from config import PATH_FB15K, PATH_MODEL_FB15K
from data_loader import DataLoader
from log_wrapper import LogWrapper
from download import download_triples
from data_creator import SentencePartialFactRelation
from upload import store_triples
from train import BaseTripletSbertModel
import torch
from evaluator import RelationPredictionEvaluator


def run_fb15():

    DATA_NAME = 'fb15k_march30_n5'
    MODEL_N = 'fb15k_batch128_ep_4'
    #####
    logger = LogWrapper(name=DATA_NAME+'_'+MODEL_N)
    gd_logger = logger.logger

    gd_logger.info(f'using gpu: {torch.cuda.is_available()}')
    gd_logger.info(f'number gpu: {torch.cuda.device_count()}')
    #####
    gd_logger.info("Data being Loaded...")
    fb15k = DataLoader(PATH_FB15K)
    fb15k.start(enrich_train=False)
    gd_logger.info("Data Loaded.")
    gd_logger.info(f"Number of unique properties in Fb15K dataset is: {len(fb15k.train_data.property.unique())}")
    try:
        partialFacts = download_triples(DATA_NAME)
        gd_logger.info("data already exists and has been retrieved from file..")

    except FileNotFoundError:

        partialFactsRel = SentencePartialFactRelation(fb15k.train_data, tn=5, thr_d_r=0.00001, min_number_props_d_r=10)
        partialFacts = partialFactsRel.create_partial_fact_relation()
        gd_logger.info(f"number of partial facts created is: {len(partialFacts)}")
        gd_logger.info("Partial Facts have been created...")
        store_triples(partialFacts, DATA_NAME)
        gd_logger.info("Partial Facts have been saved...")

    #####
    base_model = 'roberta-base'
    tsm = BaseTripletSbertModel(base_model, PATH_MODEL_FB15K, MODEL_N)
    tsm.init_model(dense_last_layer=False)

    try:
        model = tsm.load_trained_model()
        gd_logger.info("load model from file....")

    except:
        gd_logger.info("loading failed, training model....")

        model = tsm.train(partialFacts, epoch=4, batch_size=128)  # model = tsm.load_trained_model()
        gd_logger.info("Model has been trained...")
    #####

    gd_logger.info("Evaluation in Process...")
    rpe = RelationPredictionEvaluator(fb15k, model, logger=gd_logger, K_NN=1000)
    rpe.embedd_train_clusters()
    rpe.embedd_dev_test_data()
    best_k_mode = rpe.compute_aggregates_data_embedded_c_props()

    print(best_k_mode)




run_fb15()