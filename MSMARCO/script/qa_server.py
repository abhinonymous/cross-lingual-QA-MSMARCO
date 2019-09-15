import cntk as C
import numpy as np
from polymath import PolyMath
from squad_utils import metric_max_over_ground_truths, f1_score, exact_match_score
import tsv2ctf as tsv2ctf
import os
import argparse
import importlib
import time
import json
from tqdm import tqdm



import zmq
import pickle


model_name = "pm.model"

def argument_by_name(func, name):
    found = [arg for arg in func.arguments if arg.name == name]
    if len(found) == 0:
        raise ValueError('no matching names in arguments')
    elif len(found) > 1:
        raise ValueError('multiple matching names in arguments')
    else:
        return found[0]

def create_mb_and_map(func, data_file, polymath, randomize=True, repeat=True):
    mb_source = C.io.MinibatchSource(
        C.io.CTFDeserializer(
            data_file,
            C.io.StreamDefs(
                context_g_words  = C.io.StreamDef('cgw', shape=polymath.wg_dim,     is_sparse=True),
                query_g_words    = C.io.StreamDef('qgw', shape=polymath.wg_dim,     is_sparse=True),
                context_ng_words = C.io.StreamDef('cnw', shape=polymath.wn_dim,     is_sparse=True),
                query_ng_words   = C.io.StreamDef('qnw', shape=polymath.wn_dim,     is_sparse=True),
                answer_begin     = C.io.StreamDef('ab',  shape=polymath.a_dim,      is_sparse=False),
                answer_end       = C.io.StreamDef('ae',  shape=polymath.a_dim,      is_sparse=False),
                context_chars    = C.io.StreamDef('cc',  shape=polymath.word_size,  is_sparse=False),
                query_chars      = C.io.StreamDef('qc',  shape=polymath.word_size,  is_sparse=False))),
        randomize=randomize,
        max_sweeps=C.io.INFINITELY_REPEAT if repeat else 1)

    input_map = {
        argument_by_name(func, 'cgw'): mb_source.streams.context_g_words,
        argument_by_name(func, 'qgw'): mb_source.streams.query_g_words,
        argument_by_name(func, 'cnw'): mb_source.streams.context_ng_words,
        argument_by_name(func, 'qnw'): mb_source.streams.query_ng_words,
        argument_by_name(func, 'cc' ): mb_source.streams.context_chars,
        argument_by_name(func, 'qc' ): mb_source.streams.query_chars,
        argument_by_name(func, 'ab' ): mb_source.streams.answer_begin,
        argument_by_name(func, 'ae' ): mb_source.streams.answer_end
    }
    return mb_source, input_map

def create_tsv_reader(func, tsv_file, polymath, seqs, num_workers, is_test=False, misc=None):
    with open(tsv_file, 'r', encoding='utf-8') as f:
        eof = False
        batch_count = 0
        while not(eof and (batch_count % num_workers) == 0):
            batch_count += 1
            batch={'cwids':[], 'qwids':[], 'baidx':[], 'eaidx':[], 'ccids':[], 'qcids':[]}

            while not eof and len(batch['cwids']) < seqs:
                line = f.readline()
                if not line:
                    eof = True
                    break

                if misc is not None:
                    import re
                    misc['uid'].append(re.match('^([^\t]*)', line).groups()[0])

                ctokens, qtokens, atokens, cwids, qwids, baidx, eaidx, ccids, qcids = tsv2ctf.tsv_iter(line, polymath.vocab, polymath.chars, polymath.wg_dim, is_test, misc)

                batch['cwids'].append(cwids)
                batch['qwids'].append(qwids)
                batch['baidx'].append(baidx)
                batch['eaidx'].append(eaidx)
                batch['ccids'].append(ccids)
                batch['qcids'].append(qcids)

            if len(batch['cwids']) > 0:
                context_g_words  = C.Value.one_hot([[C.Value.ONE_HOT_SKIP if i >= polymath.wg_dim else i for i in cwids] for cwids in batch['cwids']], polymath.wg_dim)
                context_ng_words = C.Value.one_hot([[C.Value.ONE_HOT_SKIP if i < polymath.wg_dim else i - polymath.wg_dim for i in cwids] for cwids in batch['cwids']], polymath.wn_dim)
                query_g_words    = C.Value.one_hot([[C.Value.ONE_HOT_SKIP if i >= polymath.wg_dim else i for i in qwids] for qwids in batch['qwids']], polymath.wg_dim)
                query_ng_words   = C.Value.one_hot([[C.Value.ONE_HOT_SKIP if i < polymath.wg_dim else i - polymath.wg_dim for i in qwids] for qwids in batch['qwids']], polymath.wn_dim)
                context_chars = [np.asarray([[[c for c in cc+[0]*max(0,polymath.word_size-len(cc))]] for cc in ccid], dtype=np.float32) for ccid in batch['ccids']]
                query_chars   = [np.asarray([[[c for c in qc+[0]*max(0,polymath.word_size-len(qc))]] for qc in qcid], dtype=np.float32) for qcid in batch['qcids']]
                answer_begin = [np.asarray(ab, dtype=np.float32) for ab in batch['baidx']]
                answer_end   = [np.asarray(ae, dtype=np.float32) for ae in batch['eaidx']]

                yield { argument_by_name(func, 'cgw'): context_g_words,
                        argument_by_name(func, 'qgw'): query_g_words,
                        argument_by_name(func, 'cnw'): context_ng_words,
                        argument_by_name(func, 'qnw'): query_ng_words,
                        argument_by_name(func, 'cc' ): context_chars,
                        argument_by_name(func, 'qc' ): query_chars,
                        argument_by_name(func, 'ab' ): answer_begin,
                        argument_by_name(func, 'ae' ): answer_end }
            else:
                yield {} # need to generate empty batch for distributed training

def train(data_path, model_path, log_file, config_file, restore=False, profiling=False, gen_heartbeat=False):
    polymath = PolyMath(config_file)
    z, loss = polymath.model()
    training_config = importlib.import_module(config_file).training_config

    minibatch_size = training_config['minibatch_size']
    max_epochs = training_config['max_epochs']
    epoch_size = training_config['epoch_size']
    log_freq = training_config['log_freq']

    progress_writers = [C.logging.ProgressPrinter(
                            num_epochs = max_epochs,
                            freq = log_freq,
                            tag = 'Training',
                            log_to_file = log_file,
                            rank = C.Communicator.rank(),
                            gen_heartbeat = gen_heartbeat)]

    lr = C.learning_parameter_schedule(training_config['lr'], minibatch_size=None, epoch_size=None)

    ema = {}
    dummies = []
    for p in z.parameters:
        ema_p = C.constant(0, shape=p.shape, dtype=p.dtype, name='ema_%s' % p.uid)
        ema[p.uid] = ema_p
        dummies.append(C.reduce_sum(C.assign(ema_p, 0.999 * ema_p + 0.001 * p)))
    dummy = C.combine(dummies)

    learner = C.adadelta(z.parameters, lr)

    if C.Communicator.num_workers() > 1:
        learner = C.data_parallel_distributed_learner(learner, num_quantization_bits=1)

    trainer = C.Trainer(z, (loss, None), learner, progress_writers)

    if profiling:
        C.debugging.start_profiler(sync_gpu=True)

    train_data_file = os.path.join(data_path, training_config['train_data'])
    train_data_ext = os.path.splitext(train_data_file)[-1].lower()

    model_file = os.path.join(model_path, model_name)
    model = C.combine(list(z.outputs) + [loss.output])
    label_ab = argument_by_name(loss, 'ab')

    epoch_stat = {
        'best_val_err' : 100,
        'best_since'   : 0,
        'val_since'    : 0}

    if restore and os.path.isfile(model_file):
        trainer.restore_from_checkpoint(model_file)
        epoch_stat['best_val_err'] = validate_model(os.path.join(data_path, training_config['val_data']), model, polymath)

    def post_epoch_work(epoch_stat):
        trainer.summarize_training_progress()
        epoch_stat['val_since'] += 1

        if epoch_stat['val_since'] == training_config['val_interval']:
            epoch_stat['val_since'] = 0
            temp = dict((p.uid, p.value) for p in z.parameters)
            for p in trainer.model.parameters:
                p.value = ema[p.uid].value
            val_err = validate_model(os.path.join(data_path, training_config['val_data']), model, polymath)
            if epoch_stat['best_val_err'] > val_err:
                epoch_stat['best_val_err'] = val_err
                epoch_stat['best_since'] = 0
                trainer.save_checkpoint(model_file)
                for p in trainer.model.parameters:
                    p.value = temp[p.uid]
            else:
                epoch_stat['best_since'] += 1
                if epoch_stat['best_since'] > training_config['stop_after']:
                    return False

        if profiling:
            C.debugging.enable_profiler()

        return True

    if train_data_ext == '.ctf':
        mb_source, input_map = create_mb_and_map(loss, train_data_file, polymath)


        for epoch in range(max_epochs):
            num_seq = 0
            with tqdm(total=epoch_size, ncols=32, smoothing=0.1) as progress_bar:
                while True:
                    if trainer.total_number_of_samples_seen >= training_config['distributed_after']:
                        data = mb_source.next_minibatch(
                            minibatch_size * C.Communicator.num_workers(), input_map=input_map, 
                            num_data_partitions = C.Communicator.num_workers(), 
                            partition_index = C.Communicator.rank())
                    else:
                        data = mb_source.next_minibatch(minibatch_size, input_map=input_map)

                    trainer.train_minibatch(data)
                    num_seq += trainer.previous_minibatch_sample_count
                    dummy.eval()
                    if num_seq >= epoch_size:
                        break
                    else:
                        progress_bar.update(trainer.previous_minibatch_sample_count)
                if not post_epoch_work(epoch_stat):
                    break
    else:
        if train_data_ext != '.tsv':
            raise Exception("Unsupported format")

        minibatch_seqs = training_config['minibatch_seqs'] # number of sequences

        for epoch in range(max_epochs):       # loop over epochs
            tsv_reader = create_tsv_reader(loss, train_data_file, polymath, minibatch_seqs, C.Communicator.num_workers())
            minibatch_count = 0
            for data in tsv_reader:
                if (minibatch_count % C.Communicator.num_workers()) == C.Communicator.rank():
                    trainer.train_minibatch(data) # update model with it
                    dummy.eval()
                minibatch_count += 1
            if not post_epoch_work(epoch_stat):
                break

    if profiling:
        C.debugging.stop_profiler()

def symbolic_best_span(begin, end):
    running_max_begin = C.layers.Recurrence(C.element_max, initial_state=-float("inf"))(begin)
    return C.layers.Fold(C.element_max, initial_state=C.constant(-1e+30))(running_max_begin + end)

def validate_model(test_data, model, polymath):
    begin_logits = model.outputs[0]
    end_logits   = model.outputs[1]
    loss         = model.outputs[2]
    root = C.as_composite(loss.owner)
    mb_source, input_map = create_mb_and_map(root, test_data, polymath, randomize=False, repeat=False)
    begin_label = argument_by_name(root, 'ab')
    end_label   = argument_by_name(root, 'ae')

    begin_prediction = C.sequence.input_variable(1, sequence_axis=begin_label.dynamic_axes[1], needs_gradient=True)
    end_prediction = C.sequence.input_variable(1, sequence_axis=end_label.dynamic_axes[1], needs_gradient=True)

    best_span_score = symbolic_best_span(begin_prediction, end_prediction)
    predicted_span = C.layers.Recurrence(C.plus)(begin_prediction - C.sequence.past_value(end_prediction))
    true_span = C.layers.Recurrence(C.plus)(begin_label - C.sequence.past_value(end_label))
    common_span = C.element_min(predicted_span, true_span)
    begin_match = C.sequence.reduce_sum(C.element_min(begin_prediction, begin_label))
    end_match = C.sequence.reduce_sum(C.element_min(end_prediction, end_label))

    predicted_len = C.sequence.reduce_sum(predicted_span)
    true_len = C.sequence.reduce_sum(true_span)
    common_len = C.sequence.reduce_sum(common_span)
    f1 = 2*common_len/(predicted_len+true_len)
    exact_match = C.element_min(begin_match, end_match)
    precision = common_len/predicted_len
    recall = common_len/true_len
    overlap = C.greater(common_len, 0)
    s = lambda x: C.reduce_sum(x, axis=C.Axis.all_axes())
    stats = C.splice(s(f1), s(exact_match), s(precision), s(recall), s(overlap), s(begin_match), s(end_match))

    # Evaluation parameters
    minibatch_size = 2048
    num_sequences = 0

    stat_sum = 0
    loss_sum = 0
    
    with tqdm(ncols=32) as progress_bar:
        while True:
            data = mb_source.next_minibatch(minibatch_size, input_map=input_map)
            if not data or not (begin_label in data) or data[begin_label].num_sequences == 0:
                break
            out = model.eval(data, outputs=[begin_logits,end_logits,loss], as_numpy=False)
            testloss = out[loss]
            g = best_span_score.grad({begin_prediction:out[begin_logits], end_prediction:out[end_logits]}, wrt=[begin_prediction,end_prediction], as_numpy=False)
            other_input_map = {begin_prediction: g[begin_prediction], end_prediction: g[end_prediction], begin_label: data[begin_label], end_label: data[end_label]}
            stat_sum += stats.eval((other_input_map))
            loss_sum += np.sum(testloss.asarray())
            num_sequences += data[begin_label].num_sequences
            progress_bar.update(data[begin_label].num_sequences)

    stat_avg = stat_sum / num_sequences
    loss_avg = loss_sum / num_sequences

    print("\nValidated {} sequences, loss {:.4f}, F1 {:.4f}, EM {:.4f}, precision {:4f}, recall {:4f} hasOverlap {:4f}, start_match {:4f}, end_match {:4f}".format(
            num_sequences,
            loss_avg,
            stat_avg[0],
            stat_avg[1],
            stat_avg[2],
            stat_avg[3],
            stat_avg[4],
            stat_avg[5],
            stat_avg[6]))

    return loss_avg

# map from token to char offset
def w2c_map(s, words):
    w2c=[]
    rem=s
    offset=0
    for i, w in enumerate(words):
        cidx=rem.find(w)
        assert(cidx>=0)
        w2c.append(cidx+offset)
        offset+=cidx + len(w)
        rem=rem[cidx + len(w):]
    return w2c

# get phrase from string based on tokens and their offsets
def get_answer(raw_text, tokens, start, end):
    try:
        w2c=w2c_map(raw_text, tokens)
        return raw_text[w2c[start]:w2c[end]+len(tokens[end])]
    except:
        print(raw_text, tokens, start, end)
        import pdb
        pdb.set_trace()

def streaming_create_tsv_reader(func, line, polymath, seqs, num_workers, is_test=False, misc=None):
    ctokens, qtokens, atokens, cwids, qwids, baidx, eaidx, ccids, qcids = tsv2ctf.tsv_iter(line, polymath.vocab, polymath.chars, polymath.wg_dim, is_test, misc)
    batch = {'cwids': [], 'qwids': [], 'baidx': [], 'eaidx': [], 'ccids': [], 'qcids': []}
    batch['cwids'].append(cwids)
    batch['qwids'].append(qwids)
    batch['baidx'].append(baidx)
    batch['eaidx'].append(eaidx)
    batch['ccids'].append(ccids)
    batch['qcids'].append(qcids)


    if len(batch['cwids']) > 0:
        context_g_words = C.Value.one_hot(
            [[C.Value.ONE_HOT_SKIP if i >= polymath.wg_dim else i for i in cwids] for cwids in batch['cwids']],
            polymath.wg_dim)
        context_ng_words = C.Value.one_hot(
            [[C.Value.ONE_HOT_SKIP if i < polymath.wg_dim else i - polymath.wg_dim for i in cwids] for cwids in
             batch['cwids']], polymath.wn_dim)
        query_g_words = C.Value.one_hot(
            [[C.Value.ONE_HOT_SKIP if i >= polymath.wg_dim else i for i in qwids] for qwids in batch['qwids']],
            polymath.wg_dim)
        query_ng_words = C.Value.one_hot(
            [[C.Value.ONE_HOT_SKIP if i < polymath.wg_dim else i - polymath.wg_dim for i in qwids] for qwids in
             batch['qwids']], polymath.wn_dim)
        context_chars = [
            np.asarray([[[c for c in cc + [0] * max(0, polymath.word_size - len(cc))]] for cc in ccid], dtype=np.float32)
            for ccid in batch['ccids']]
        query_chars = [
            np.asarray([[[c for c in qc + [0] * max(0, polymath.word_size - len(qc))]] for qc in qcid], dtype=np.float32)
            for qcid in batch['qcids']]
        answer_begin = [np.asarray(ab, dtype=np.float32) for ab in batch['baidx']]
        answer_end = [np.asarray(ae, dtype=np.float32) for ae in batch['eaidx']]

        return {argument_by_name(func, 'cgw'): context_g_words,
               argument_by_name(func, 'qgw'): query_g_words,
               argument_by_name(func, 'cnw'): context_ng_words,
               argument_by_name(func, 'qnw'): query_ng_words,
               argument_by_name(func, 'cc'): context_chars,
               argument_by_name(func, 'qc'): query_chars,
               argument_by_name(func, 'ab'): answer_begin,
               argument_by_name(func, 'ae'): answer_end}



def streaming_inference(model_path, model_file, config_file,port = "8889", is_test = 1):
    polymath = PolyMath(config_file)
    model = C.load_model(os.path.join(model_path, model_file if model_file else model_name))
    begin_logits = model.outputs[0]
    end_logits = model.outputs[1]
    loss = C.as_composite(model.outputs[2].owner)
    begin_prediction = C.sequence.input_variable(1, sequence_axis=begin_logits.dynamic_axes[1], needs_gradient=True)
    end_prediction = C.sequence.input_variable(1, sequence_axis=end_logits.dynamic_axes[1], needs_gradient=True)
    best_span_score = symbolic_best_span(begin_prediction, end_prediction)
    predicted_span = C.layers.Recurrence(C.plus)(begin_prediction - C.sequence.past_value(end_prediction))

    batch_size = 1  # in sequences
    misc = {'rawctx': [], 'ctoken': [], 'answer': [], 'uid': []}
    Flag = True

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:8889")

    while True :
        message = socket.recv()
        question_str, context_str = pickle.loads(message)

        line = "1102432\tDESCRIPTION\t" + context_str +"\t"+question_str
        data = streaming_create_tsv_reader(loss, line, polymath, batch_size, 1, is_test=True, misc=misc)
        out = model.eval(data, outputs=[begin_logits, end_logits, loss], as_numpy=False)
        g = best_span_score.grad({begin_prediction: out[begin_logits], end_prediction: out[end_logits]},
                                 wrt=[begin_prediction, end_prediction], as_numpy=False)
        other_input_map = {begin_prediction: g[begin_prediction], end_prediction: g[end_prediction]}
        span = predicted_span.eval((other_input_map))
        #print("just before for {}".format(misc['ctoken']))
        seq, raw_text, ctokens, answer, uid = 0, misc['rawctx'], misc['ctoken'], misc['answer'], misc['uid']
        #print("just AFTER for {}".format(ctokens))
        seq_where = np.argwhere(span[seq])[:, 0]
        span_begin = np.min(seq_where)
        span_end = np.max(seq_where)
        #print("before predict")
        predict_answer = get_answer(raw_text[0], ctokens[0], span_begin, span_end)
        # results['query_id'] = int(uid)
        result = (question_str,predict_answer)
        socket.send(pickle.dumps(result))

def test(test_data, model_path, model_file, config_file):
    polymath = PolyMath(config_file)
    model = C.load_model(os.path.join(model_path, model_file if model_file else model_name))
    begin_logits = model.outputs[0]
    end_logits   = model.outputs[1]
    loss         = C.as_composite(model.outputs[2].owner)
    begin_prediction = C.sequence.input_variable(1, sequence_axis=begin_logits.dynamic_axes[1], needs_gradient=True)
    end_prediction = C.sequence.input_variable(1, sequence_axis=end_logits.dynamic_axes[1], needs_gradient=True)
    best_span_score = symbolic_best_span(begin_prediction, end_prediction)
    predicted_span = C.layers.Recurrence(C.plus)(begin_prediction - C.sequence.past_value(end_prediction))

    with open(test_data) as f:
        num_test = sum(1 for line in f)

    batch_size = 16 # in sequences
    misc = {'rawctx':[], 'ctoken':[], 'answer':[], 'uid':[]}
    tsv_reader = create_tsv_reader(loss, test_data, polymath, batch_size, 1, is_test=True, misc=misc)
    results = {}

    with tqdm(total=num_test, ncols=32) as progress_bar:
        with open('{}_out.json'.format(model_file), 'w', encoding='utf-8') as json_output:
            for data in tsv_reader:
                out = model.eval(data, outputs=[begin_logits, end_logits, loss], as_numpy=False)
                g = best_span_score.grad({begin_prediction: out[begin_logits], end_prediction: out[end_logits]}, 
                    wrt=[begin_prediction,end_prediction], as_numpy=False)
                other_input_map = {begin_prediction: g[begin_prediction], end_prediction: g[end_prediction]}
                span = predicted_span.eval((other_input_map))
                for seq, (raw_text, ctokens, answer, uid) in enumerate(
                    zip(misc['rawctx'], misc['ctoken'], misc['answer'], misc['uid'])):
                    seq_where = np.argwhere(span[seq])[:, 0]
                    span_begin = np.min(seq_where)
                    span_end = np.max(seq_where)
                    predict_answer = get_answer(raw_text, ctokens, span_begin, span_end)
                    results['query_id'] = int(uid)
                    results['answers'] = [predict_answer]
                    json.dump(results, json_output)
                    json_output.write("\n")

                misc['rawctx'] = []
                misc['ctoken'] = []
                misc['answer'] = []
                misc['uid'] = []
                progress_bar.update(batch_size)

if __name__=='__main__':
    # default Paths relative to current python file.
    abs_path   = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(abs_path, '.')
    data_path  = os.path.join(abs_path, '.')

    parser = argparse.ArgumentParser()
    parser.add_argument('-datadir', '--datadir', help='Data directory where the dataset is located', required=False, default=data_path)
    parser.add_argument('-outputdir', '--outputdir', help='Output directory for checkpoints and models', required=False, default=None)
    parser.add_argument('-logdir', '--logdir', help='Log file', required=False, default=None)
    parser.add_argument('-profile', '--profile', help="Turn on profiling", action='store_true', default=False)
    parser.add_argument('-genheartbeat', '--genheartbeat', help="Turn on heart-beat for philly", action='store_true', default=False)
    parser.add_argument('-config', '--config', help='Config file', required=False, default='config')
    parser.add_argument('-r', '--restart', help='Indicating whether to restart from scratch (instead of restart from checkpoint file by default)', action='store_true')
    parser.add_argument('-test', '--test', help='Test data file', required=False, default=None)
    parser.add_argument('-model', '--model', help='Model file name', required=False, default=model_name)

    args = vars(parser.parse_args())
    streaming_str = "A company is incorporated in a specific nation , often within the bounds of a smaller subset of that nation , such as a state or province . The corporation is then governed by the laws of incorporation in that state . A corporation may issue stock, either private or public , or may be classified as a non - stock corporation . If stock is issued , the corporation will usually be governed by its shareholders , either directly or indirectly . Today , there is a growing community of more than 2,100 Certified B Corps from 50 countries and over 130 industries working together toward 1 unifying goal : to redefine success in business . Join the Movement Corporation definition , an association of individuals , created by law or under authority of law , having a continuous existence independent of the existences of its members , and powers and liabilities distinct from those of its members . See more . Examples of corporation in a Sentence . 1 He works as a consultant for several large corporations . 2 a substantial corporation that showed that he was a sucker for all - you - can - eat buffets . 1 : a government - owned corporation ( as a utility or railroad ) engaged in a profit - making enterprise that may require the exercise of powers unique to government ( as eminent domain ) — called also government corporation , publicly held corporation McDonald 's Corporation is one of the most recognizable corporations in the world . A corporation is a company or group of people authorized to act as a single entity ( legally a person ) and recognized as such in law . Early incorporated entities were established by charter ( i.e . by an ad hoc act granted by a monarch or passed by a parliament or legislature ) . Corporations are owned by their stockholders ( shareholders ) who share in profits and losses generated through the firm 's operations , and have three distinct characteristics ( 1 ) Legal existence : a firm can ( like a person ) buy , sell , own , enter into a contract , and sue other persons and firms , and be sued by them . An Association is an organized group of people who share in a common interest , activity , or purpose . 1 Start a business Plan your business . Create your business structure Types of business structures . 2 Change or update your business Add a new location to your existing business . Add an endorsement to your existing business . B Corp certification shines a light on the companies leading the global movement ... LLCs offer greater flexibility when it comes to income taxes . 1 The owner or member of an LLC can have their income taxed in three ways : 2 A single owner LLC is treated as a Schedule C ( sole proprietor ) for tax purposes ."
    streaming_str =streaming_str*100
    streaming_str = "1102432\tDESCRIPTION\t" + streaming_str + "\twhat is a corporation ?"
    # test_data = args['test']
    model = args['model']
    # if test_data:
    #     test(test_data, model_path, test_model, args['config'])
    # else:
    #     try:
    #         train(data_path, model_path, args['logdir'], args['config'],
    #             restore = not args['restart'],
    #             profiling = args['profile'],
    #             gen_heartbeat = args['genheartbeat'])
    #     finally:
    #         C.Communicator.finalize()

    print("starting streaming mode")
    # streaming_inference(streaming_str, model_path, model, args['config'])
    streaming_inference(model_path, model, args['config'])


