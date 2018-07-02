import autograd as ag
import copy
import numpy as np
import logging
import pickle

from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import RobustScaler
from sklearn.utils import check_random_state

from recnn.preprocessing import rewrite_content
from recnn.preprocessing import permute_by_pt
from recnn.preprocessing import extract
from recnn.preprocessing import multithreadmap


from recnn.recnn import log_loss
from recnn.recnn import square_error
from recnn.recnn import adam
from recnn.recnn import grnn_init_simple
from recnn.recnn import grnn_predict_simple
from recnn.recnn import grnn_init_gated
from recnn.recnn import grnn_predict_gated


logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s %(levelname)s] %(message)s")

def tftransform(jet,tf) :
    jet["content"] = tf.transform(jet["content"])
    return(jet)

def train(filename_train,
          filename_model,
          regression=False,
          n_events_train=-1,
          simple=False,
          n_features=12,
          n_hidden=40,
          n_epochs=5,
          batch_size=64,
          step_size=0.0005,
          decay=0.9,
          random_state=42,
          verbose=False,
          statlimit=-1):
    # Initialization
    gated = not simple
    if verbose:
        logging.info("Calling with...")
        logging.info("\tfilename_train = %s" % filename_train)
        logging.info("\tfilename_model = %s" % filename_model)
        logging.info("\tn_events_train = %d" % n_events_train)
        logging.info("\tgated = %s" % gated)
        logging.info("\tn_features = %d" % n_features)
        logging.info("\tn_hidden = %d" % n_hidden)
        logging.info("\tn_epochs = %d" % n_epochs)
        logging.info("\tbatch_size = %d" % batch_size)
        logging.info("\tstep_size = %f" % step_size)
        logging.info("\tdecay = %f" % decay)
        logging.info("\trandom_state = %d" % random_state)
    rng = check_random_state(random_state)

    # Make data
    if verbose:
        logging.info("Loading data...")
    if filename_train[-1]=="e":
        fd = open(filename_train, "rb")
        X, y = pickle.load(fd)
        fd.close()
    else:
        X, y = np.load(filename_train)
    X = np.array(X).astype(dict)[:statlimit]
    y = np.array(y).astype(float)[:statlimit]

    if regression:
	    y_pred_0 = [x["pt"] for x in X]
	    zerovalue=square_error(y, y_pred_0).mean()

    if n_events_train > 0:
        indices = check_random_state(123).permutation(len(X))[:n_events_train]
        X = X[indices]
        y = y[indices]
    X = list(X)
    if verbose:
        logging.info("\tfilename = %s" % filename_train)
        logging.info("\tX size = %d" % len(X))
        logging.info("\ty size = %d" % len(y))

    # Preprocessing
    if verbose:
        logging.info("Preprocessing...")
    X=multithreadmap(rewrite_content,X)
    X=multithreadmap(permute_by_pt,X)
    X = multithreadmap(extract,X)
    tf = RobustScaler().fit(np.vstack([jet["content"] for jet in X]))

    X = multithreadmap(tftransform,X,tf=tf)

    # Split into train+validation
    logging.info("Splitting into train and validation...")

    X_train, X_valid, y_train, y_valid = train_test_split(X, y,
                                                          test_size=0.1,
                                                          random_state=rng)

    # Training
    if verbose:
        logging.info("Training...")

    if gated:
        predict = grnn_predict_gated
        init = grnn_init_gated
    else:
        predict = grnn_predict_simple
        init = grnn_init_simple

    trained_params = init(n_features, n_hidden, random_state=rng)
    n_batches = int(np.ceil(len(X_train) / batch_size))
    best_score = [np.inf]  # yuck, but works
    best_params = [trained_params]

    def loss(X, y, params):
        y_pred = predict(params, X, regression=regression)
        if regression:
            l = square_error(y, y_pred).mean()
        else :
            l = log_loss(y, y_pred).mean()
        return l

    def objective(params, iteration):
        rng = check_random_state(iteration % n_batches)
        start = rng.randint(len(X_train) - batch_size)
        idx = slice(start, start+batch_size)
        return loss(X_train[idx], y_train[idx], params)

    def callback(params, iteration, gradient):
        if iteration % 100 == 0:
            the_loss = loss(X_valid, y_valid, params)
            if the_loss < best_score[0]:
                best_score[0] = the_loss
                best_params[0] = copy.deepcopy(params)

                fd = open(filename_model, "wb")
                pickle.dump(best_params[0], fd)
                fd.close()

            if verbose:
                if regression :
                    logging.info(
                        "%5d\t~loss(train)=%.4f\tloss(valid)=%.4f"
                        "\tbest_loss(valid)=%.4f" % (
                            iteration,
                            loss(X_train[:5000], y_train[:5000], params),
                            loss(X_valid, y_valid, params),
                            best_score[0]))
                else:
                    roc_auc = roc_auc_score(y_valid, predict(params, X_valid,regression=regression))
                    logging.info(
                        "%5d\t~loss(train)=%.4f\tloss(valid)=%.4f"
                        "\troc_auc(valid)=%.4f\tbest_loss(valid)=%.4f" % (
                            iteration,
                            loss(X_train[:5000], y_train[:5000], params),
                            loss(X_valid, y_valid, params),
                            roc_auc,
                            best_score[0]))


    for i in range(n_epochs):
        logging.info("epoch = %d" % i)
        logging.info("step_size = %.4f" % step_size)
        if regression:
            logging.info("zerovalue = %.4f" % zerovalue)

        trained_params = adam(ag.grad(objective),
                              trained_params,
                              step_size=step_size,
                              num_iters=1 * n_batches,
                              callback=callback)
        step_size = step_size * decay


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='lol')
    parser.add_argument("filename_train", help="",type=str)
    parser.add_argument("filename_model", help="", type=str)
    parser.add_argument("--regression", help="", action="store_true")
    parser.add_argument("--n_events_train", help="", type=int, default=-1)
    parser.add_argument("--n_features", help="", type=int, default=12)
    parser.add_argument("--n_hidden", help="", type=int, default=40)
    parser.add_argument("--n_epochs", help="", type=int, default=5)
    parser.add_argument("--batch_size", help="", type=int, default=64)
    parser.add_argument("--step_size", help="", type=float, default=0.0005)
    parser.add_argument("--decay", help="", type=float, default=0.9)
    parser.add_argument("--random_state", help="", type=int, default=42)
    parser.add_argument("--statlimit", help="", type=int, default=-1)
    parser.add_argument("--verbose", help="", action="store_true")
    parser.add_argument("--simple", help="", action="store_true")
    args = parser.parse_args()
    
    train(filename_train=args.filename_train,
          filename_model=args.filename_model,
          regression=args.regression,
          n_events_train=args.n_events_train,
          simple=args.simple,
          n_features=args.n_features,
          n_hidden=args.n_hidden,
          n_epochs=args.n_epochs,
          batch_size=args.batch_size,
          step_size=args.step_size,
          decay=args.decay,
          random_state=args.random_state,
          verbose=args.verbose,
          statlimit=args.statlimit)
