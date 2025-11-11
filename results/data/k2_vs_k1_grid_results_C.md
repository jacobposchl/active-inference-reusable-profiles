# Heatmap analysis — Case C

File: `C:\Users\Jacob Poschl\reusable profiles in active inference\active-inference-reusable-profiles\results\data\k2_vs_k1_grid_results_C.npz`

## Contents

Found keys: ['prob_rewards', 'prob_hints', 'll_k1', 'll_k2', 'acc_k1', 'acc_k2', 'bic_k1', 'bic_k2', 'll_delta', 'acc_delta', 'bic_delta']

- `prob_rewards`: [0.65, 0.75, 0.85, 0.95]

- `prob_hints`: [0.65, 0.75, 0.85, 0.95]

- prob_rewards: dtype=float64, shape=(4,)
  - min=0.65, mean=0.8, max=0.95, std=0.111803

- prob_hints: dtype=float64, shape=(4,)
  - min=0.65, mean=0.8, max=0.95, std=0.111803

- ll_k1: dtype=float64, shape=(4, 4)
  - min=-0.753297, mean=-0.554421, max=-0.37607, std=0.11239
  - sample values:
    [-0.591935809167117, -0.5265492982162232, -0.47082932931290417, -0.41791349356526425]
    [-0.7532973778649167, -0.6677195246046803, -0.5905682060650188, -0.5204173259107483]
    [-0.6114428103461379, -0.6664455809079277, -0.6627979900498946, -0.6898598979978978]
    [-0.37607049208732224, -0.3990005239073988, -0.44225789861505277, -0.48363477868119603]

- ll_k2: dtype=float64, shape=(4, 4)
  - min=-0.226885, mean=-0.0862328, max=-0.0272678, std=0.0536402
  - sample values:
    [-0.22688538529058586, -0.16828361845519113, -0.1374989391293471, -0.12231208984553457]
    [-0.10607061547812759, -0.10161592870766208, -0.08888098205031948, -0.06753847479537403]
    [-0.027267821275132553, -0.031595378579372425, -0.042724836023255816, -0.05471165182808957]
    [-0.0482028319422039, -0.05302381652221918, -0.0638367016127508, -0.03927546979911149]

- acc_k1: dtype=float64, shape=(4, 4)
  - min=0, mean=0.406039, max=0.9315, std=0.418183
  - sample values:
    [0.000125, 0.0, 0.0, 0.0]
    [0.0485, 0.027, 0.006875, 0.000625]
    [0.903375, 0.8045, 0.694375, 0.33875]
    [0.9315, 0.9315, 0.909625, 0.899875]

- acc_k2: dtype=float64, shape=(4, 4)
  - min=0.65975, mean=0.847023, max=0.927625, std=0.0790287
  - sample values:
    [0.65975, 0.722, 0.763125, 0.80175]
    [0.82125, 0.800375, 0.818875, 0.855375]
    [0.9135, 0.90925, 0.905375, 0.89875]
    [0.927625, 0.927625, 0.922, 0.90575]

- bic_k1: dtype=float64, shape=(4, 4)
  - min=6044.09, mean=8897.7, max=12079.7, std=1798.24
  - sample values:
    [9497.934537135858, 8451.750361921557, 7560.230859468453, 6713.577487506214]
    [12079.719636300653, 10710.47398413687, 9476.052887502286, 8353.63880503396]
    [9810.046556000192, 10690.09088498883, 10631.729431260299, 11064.719958428352]
    [6044.089463859142, 6410.969972980367, 7103.087968302831, 7765.118049361123]

- bic_k2: dtype=float64, shape=(4, 4)
  - min=499.196, mean=1442.63, max=3693.08, std=858.244
  - sample values:
    [3693.0765423940074, 2755.448273027692, 2262.8934038141874, 2019.9038152731869]
    [1760.0402253946752, 1688.765237067227, 1485.0060905497455, 1143.5259744706182]
    [499.1955181467547, 568.4364350145926, 746.5077541167269, 938.296806994067]
    [834.1556888198962, 911.2914421001407, 1084.2976035486465, 691.3178945304177]

- ll_delta: dtype=float64, shape=(4, 4)
  - min=0.295601, mean=0.468188, max=0.647227, std=0.124779
  - sample values:
    [0.3650504238765311, 0.3582656797610321, 0.33333039018355703, 0.29560140371972965]
    [0.6472267623867891, 0.5661035958970182, 0.5016872240146993, 0.4528788511153743]
    [0.5841749890710053, 0.6348502023285553, 0.6200731540266388, 0.6351482461698083]
    [0.32786766014511837, 0.34597670738517966, 0.378421197002302, 0.4443593088820845]

- acc_delta: dtype=float64, shape=(4, 4)
  - min=-0.003875, mean=0.440984, max=0.85475, std=0.355728
  - sample values:
    [0.6596249999999999, 0.722, 0.763125, 0.80175]
    [0.77275, 0.7733749999999999, 0.812, 0.85475]
    [0.01012499999999994, 0.10475000000000001, 0.21100000000000008, 0.56]
    [-0.003874999999999962, -0.003874999999999962, 0.012375000000000025, 0.005875000000000075]

- bic_delta: dtype=float64, shape=(4, 4)
  - min=-10319.7, mean=-7455.07, max=-4693.67, std=1996.47
  - sample values:
    [-5804.85799474185, -5696.3020888938645, -5297.337455654266, -4693.673672233028]
    [-10319.679410905977, -9021.708747069642, -7991.046796952541, -7210.112830563341]
    [-9310.851037853437, -10121.654449974236, -9885.221677143572, -10126.423151434285]
    [-5209.933775039245, -5499.678530880226, -6018.790364754184, -7073.800154830705]

Saved annotated heatmap to `C:\Users\Jacob Poschl\reusable profiles in active inference\active-inference-reusable-profiles\results\figures\k2_vs_k1_grid_results_C_ll_delta.png` and `C:\Users\Jacob Poschl\reusable profiles in active inference\active-inference-reusable-profiles\results\figures\k2_vs_k1_grid_results_C_ll_delta.pdf`

Saved annotated heatmap to `C:\Users\Jacob Poschl\reusable profiles in active inference\active-inference-reusable-profiles\results\figures\k2_vs_k1_grid_results_C_acc_delta.png` and `C:\Users\Jacob Poschl\reusable profiles in active inference\active-inference-reusable-profiles\results\figures\k2_vs_k1_grid_results_C_acc_delta.pdf`

Saved annotated heatmap to `C:\Users\Jacob Poschl\reusable profiles in active inference\active-inference-reusable-profiles\results\figures\k2_vs_k1_grid_results_C_bic_delta.png` and `C:\Users\Jacob Poschl\reusable profiles in active inference\active-inference-reusable-profiles\results\figures\k2_vs_k1_grid_results_C_bic_delta.pdf`
