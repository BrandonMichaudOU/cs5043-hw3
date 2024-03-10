import numpy as np
import matplotlib.pyplot as plt
import pickle


def figure3():
    deep_test_accuracy = np.empty(5)
    shallow_test_accuracy = np.empty(5)
    for r in range(5):
        with open(f'results/image_Csize_5_3_Cfilters_10_10_Pool_2_2_Pad_valid_hidden_50_20_LR_0.001000_ntrain_03_rot_'
                  f'{r:02d}_results.pkl', "rb") as fp:
            results = pickle.load(fp)
            shallow_test_accuracy[r] = results['predict_testing_eval'][1]
        with open(f'results/image_Csize_1_5_3_1_5_3_1_5_3_1_5_3_1_5_3_Cfilters_8_8_8_16_16_16_32_32_32_64_64_64_32_32_32'
                  f'_Pool_1_1_1_2_1_1_2_1_1_2_1_1_2_1_1_Pad_same_hidden_1024_512_256_128_64_drop_0.500_sdrop_0.200_'
                  f'L2_0.001000_LR_0.000100_ntrain_03_rot_{r:02d}_results.pkl', "rb") as fp:
            results = pickle.load(fp)
            deep_test_accuracy[r] = results['predict_testing_eval'][1]
    fig = plt.figure()
    plt.hist(shallow_test_accuracy, bins=5, color='blue', edgecolor='black', alpha=0.5, label='Shallow')
    plt.hist(deep_test_accuracy, bins=5, color='red', edgecolor='black', alpha=0.5, label='Deep')
    plt.title('Frequency of Test Accuracy')
    plt.xlabel('Test Accuracy')
    plt.ylabel('Frequency')
    plt.legend()
    fig.savefig('figures/fig3.png')


if __name__ == '__main__':
    figure3()
