import numpy as np
import os

# attack using partial channels
data_name = ['ERN', 'MI', 'P300']  # 'ERN' or 'MI' or 'P300'
model_used = {'ERN': ['EEGNet', 'DeepCNN', 'xDAWN'],
              'MI': ['EEGNet', 'DeepCNN', 'CSP'],
              'P300': ['EEGNet', 'DeepCNN', 'xDAWN']}
npp_params_list = {'ERN': [0.3, 5, 0.1], 'MI': [1.5, 5, 0.1], 'P300': [0.01, 5, 0.1]}
partials = [1.0, 0.3, 0.2, 0.1]
repeat = 10

save_dir = 'runs/attack_using_partial_channels'

for data in data_name:
    models = model_used[data]
    npp_params = npp_params_list[data]
    for model in models:
        results_dir = os.path.join(save_dir, data, model)
        print('*' * 30 + data + '-' + model + '*' * 30)
        for partial in partials:
            raccs = []
            rasrs = []
            for r in range(repeat):
                if partial == 1.0:
                    results = np.load(os.path.join('runs/physical_attack/', data, model + '/run{}/npp_{}_{}_{}.npz'.format(r, npp_params[0], npp_params[1], npp_params[2])))
                    raccs.append(results['accs'])
                    rasrs.append(results['poison_rates'])
                else:
                    results = np.load(results_dir + '/run{}/{}.npz'.format(r, partial))

                    raccs.append(results['accs'])
                    rasrs.append(results['asrs'])

            raccs, rasrs = np.array(raccs), np.array(rasrs)

            raccs, rasrs = np.mean(raccs, axis=1), np.mean(rasrs, axis=1)

            print('NPP {}: ACC: mean={}ASR: mean={}'.format(partial, np.mean(raccs), np.mean(rasrs)))
