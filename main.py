import numpy as np
from data_loder import Data_Loader
import scipy.io
import opt_tc as tc
import argparse
from data_preprossing import feature_pre_processing

# The code is adapted from  https://github.com/lironber/GOAD.git



def norm_data(train_real, val_real, val_fake):
    mus = train_real.mean(0)
    sds = train_real.std(0)
    sds[sds == 0] = 1

    def get_norm(xs, mu, sd):
        return np.array([(x - mu) / sd for x in xs])

    train_real = get_norm(train_real, mus, sds)
    val_real = get_norm(val_real, mus, sds)
    val_fake = get_norm(val_fake, mus, sds)
    return train_real, val_real, val_fake


def norm(data, mu=1):
    return 2 * (data / 255.) - mu


def Thyroid_train_valid_data():
    data = scipy.io.loadmat("data/thyroid.mat")
    samples = data['X']  # 3772
    labels = ((data['y']).astype(np.int32)).reshape(-1)

    norm_samples = samples[labels == 0]  # 3679 norm
    anom_samples = samples[labels == 1]  # 93 anom

    n_train = len(norm_samples) // 2
    x_train = norm_samples[:n_train]  # 1839 train

    val_real = norm_samples[n_train:]
    val_fake = anom_samples
    return norm_data(x_train, val_real, val_fake)


def load_trans_data():
    train_real, val_real, val_fake = Thyroid_train_valid_data()
    y_test_fscore = np.concatenate([np.zeros(len(val_real)), np.ones(len(val_fake))])
    ratio = 100.0 * len(val_real) / (len(val_real) + len(val_fake))

    n_train, n_dims = train_real.shape
    rots = np.random.randn(n_rots, n_dims, d_out)

    print('Calculating transforms')
    x_train = np.stack([train_real.dot(rot) for rot in rots], 2)
    print(train_real.shape, x_train.shape)
    val_real_xs = np.stack([val_real.dot(rot) for rot in rots], 2)
    val_fake_xs = np.stack([val_fake.dot(rot) for rot in rots], 2)
    x_test = np.concatenate([val_real_xs, val_fake_xs])
    return x_train, x_test, y_test_fscore, ratio



def train_anomaly_detector():
    x_train_T, x_test_T, y_test, ratio = load_trans_data()

    int_feat_cnt = x_train_T.shape[1]
    cate_feat_cnt = 0
    data = feature_pre_processing( np.concatenate([x_train_T, x_test_T]), int_feat_cnt, cate_feat_cnt, with_int_feat)
    x_train = data[0 : x_train_T.shape[0],:,:]
    x_test = data[x_train_T.shape[0]:,:,:]
    for i, data_one_trans in enumerate(x_train.transpose()):
        print("hello")
        # f_score = tc_obj.fit_trans_classifier(x_train, x_test, y_test, ratio)
    # return f_score


if __name__ == '__main__':
    n_rots = 5
    d_out = 32
    n_iters = 10

    with_int_feat = True
    f_scores = np.zeros(n_iters)
    for i in range(n_iters):
        f_scores[i] = train_anomaly_detector()
    print("AVG f1_score", f_scores.mean())
