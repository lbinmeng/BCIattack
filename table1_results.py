import numpy as np
import os

data_name = ['ERN', 'MI', 'P300']  # 'ERN' or 'MI' or 'P300'
model_used = {'ERN': ['EEGNet', 'DeepCNN', 'xDAWN'],
              'MI': ['EEGNet', 'DeepCNN', 'CSP'],
              'P300': ['EEGNet', 'DeepCNN', 'xDAWN']}
amp_name = ['low', 'middle', 'high']
amps = {'ERN': [0.1, 0.2, 0.3],
        'MI': [0.5, 1.0, 1.5],
        'P300': [0.001, 0.005, 0.01]}
npp_params = [0.3, 5, 0.1]
repeat = 10

save_dir = 'runs/attack_performance'

for data in data_name:
    models = model_used[data]
    amp = amps[data]
    for model in models:
        results_dir = os.path.join(save_dir, data, model)
        print('*' * 30 + data + '-' + model + '*' * 30)
        print('*' * 30 + 'NPP Baseline' + '*' * 30)
        for i in range(3):
            raccs = []
            b_rbcas = []
            rasrs = []
            for r in range(repeat):
                results = np.load(results_dir + '/run{}/baseline_{}_{}_{}.npz'.format(r, amp[i], 5, 0.1))

                raccs.append(results['accs'])
                rasrs.append(results['poison_rates'])

            raccs,  rasrs = np.array(raccs), np.array(rasrs)

            raccs,  rasrs = np.mean(raccs, axis=1), np.mean(rasrs, axis=1)
        
            print('{} amp: ACC: mean={}, ASR: mean={}'.format(amp_name[i], np.mean(raccs), np.mean(rasrs)))

        print('*' * 30 + 'NPP Attack' + '*' * 30)
        for i in range(3):
            raccs = []
            rasrs = []
            for r in range(repeat):
                results = np.load(results_dir + '/run{}/npp_{}_{}_{}.npz'.format(r, amp[i], 5, 0.1))

                raccs.append(results['accs'])
                rasrs.append(results['poison_rates'])

            raccs,  rasrs = np.array(raccs), np.array(rasrs)

            raccs,  rasrs = np.mean(raccs, axis=1), np.mean(rasrs, axis=1)
        
            print('{} amp: ACC: mean={}, ASR: mean={}'.format(amp_name[i], np.mean(raccs), np.mean(rasrs)))


