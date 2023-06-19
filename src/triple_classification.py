import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from data_loader import DataLoader
from data_creator import SentenceTripleEntity
from train import BaseTripletSbertModel, BaseTripletSbertModelSimple
from evaluator import TripleClassifierEvaluator
from log_wrapper import LogWrapper
import torch
from upload import store_triples
from download import download_triples
from config import PATH_FB13, PATH_MODEL_FB13


def run_fb13():
    DATA_NAME = 'fb13_fn5tn0_mar17'
    MODEL_N = 'fb13_batch64_ep_4'
    #####
    logger = LogWrapper(name=DATA_NAME + '_' + MODEL_N)
    gd_logger = logger.logger
    #####
    gd_logger.info(f'using gpu: {torch.cuda.is_available()}')
    gd_logger.info(f'number gpu: {torch.cuda.device_count()}')

    gd_logger.info("Data being Loaded...")
    fb13 = DataLoader(PATH_FB13)
    fb13.start(enrich_train=False)
    gd_logger.info("Data Loaded.")
    #####

    try:
        triples = download_triples(DATA_NAME)

    except FileNotFoundError:

        sentencetripleent = SentenceTripleEntity(fb13.train_data, entity='subject', fn=5, tn=0, logger=gd_logger,
                                                 rnd_spo_bool=True)
        triples = sentencetripleent.create_sentences_triple()
        gd_logger.info("Partial Facts have been created...")
        store_triples(triples, DATA_NAME)
        gd_logger.info("Partial Facts have been saved...")

    #####
    base_model = 'roberta-base'

    tsm = BaseTripletSbertModel(base_model, PATH_MODEL_FB13, MODEL_N)
    tsm.init_model(dense_last_layer=False)

    try:
        model = tsm.load_trained_model()
        gd_logger.info("load model from file....")

    except:
        gd_logger.info("loading failed, training model....")

        model = tsm.train(triples, epoch=4, batch_size=64)  # model = tsm.load_trained_model()
        gd_logger.info("Model has been trained...")
    ######
    '''

    gd_logger.info("BaseTripletSbertModelSimple...!")

    tsm = BaseTripletSbertModelSimple(base_model, PATH_MODEL_FB13, MODEL_N)
    model = tsm.train(triples, epoch=4, batch_size=128)
    '''
    gd_logger.info("Evaluation in Process...")
    tce = TripleClassifierEvaluator(fb13, model, logger=gd_logger)
    tce.embedd_train_clusters()
    tce.embedd_dev_test_data()
    tce.compute_dist_d_t_to_ent_embedds_faiss()
    tce.find_thr_dev_train()
    tce.evaluate_on_test()

    gd_logger.info("END OF LOG")


run_fb13()
