import torch
from dataset.DatasetType import DatasetType
from GQN.nn.cnn.RepresentationNNTypes import RepresentationNNTypes


class Properties:
    def __init__(self,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                 dataset_type=DatasetType.DEEPMIND,
                 data_path=None,
                 json_path=None,
                 deepmind_dataset=None,
                 deepmind_dataset_context_size=9,
                 deepmind_dataset_root_path=None,
                 representation=RepresentationNNTypes.TOWER,

                 L=12,
                 B=32,
                 s_max=100000,
                 mi_I=5 * 10 ** (-4),
                 mi_F=5 * 10 ** (-5),
                 mi_N=1.6 * 10 ** 6,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=10 ** (-8),
                 sigma_I=2.0,
                 sigma_F=0.7,
                 sigma_N=2 * 10 ** 5,
                 H_g_depth=64,  # every property from H_g_depth to C_e_depth must have this same value to work properly
                 C_g_depth=64,
                 U_depth=64,
                 H_e_depth=64,
                 C_e_depth=64,
                 R_depth=256,
                 X_depth=3,
                 V_depth=7,
                 Z_depth=3,
                 test_interval=20,
                 generated_images_path="results/generated_images",
                 generated_images2_path="results/generated_images2",
                 generated_images3_path="results/generated_images3",
                 representation_images_path="results/representation_images",
                 referenced_images_path="results/referenced_images",
                 log_file_path="results/logs/results.log",
                 save_images=True,
                 save_model_interval=250,
                 save_model_path="results/models"
                 ):
        self.device = device

        self.representation = representation

        self.dataset_type = dataset_type
        self.deepmind_dataset = deepmind_dataset
        self.deepmind_dataset_context_size = deepmind_dataset_context_size
        self.deepmind_dataset_root_path = deepmind_dataset_root_path
        self.data_path = data_path
        self.json_path = json_path

        self.image_resize = (64, 64)  # TODO make adjustable by constructor (check if possible)

        self.L = L
        self.B = B
        self.s_max = s_max  # 2000000

        self.mi_I = mi_I
        self.mi_F = mi_F
        self.mi_N = mi_N

        self.beta_1 = beta_1
        self.beta_2 = beta_2

        self.epsilon = epsilon

        self.sigma_I = sigma_I
        self.sigma_F = sigma_F
        self.sigma_N = sigma_N

        self.R_dim = 16 if RepresentationNNTypes.TOWER == representation else 1

        self.H_g_depth = H_g_depth
        self.C_g_depth = C_g_depth
        self.U_depth = U_depth

        self.H_e_depth = H_e_depth
        self.C_e_depth = C_e_depth

        self.R_depth = R_depth

        self.X_depth = X_depth
        self.V_depth = V_depth
        self.Z_depth = Z_depth

        self.Z_dim = 16  # TODO make adjustable by constructor (check if possible)

        self.test_interval = test_interval
        self.generated_images_path = generated_images_path
        self.generated_images2_path = generated_images2_path
        self.generated_images3_path = generated_images3_path
        self.representation_images_path = representation_images_path
        self.referenced_images_path = referenced_images_path
        self.log_file_path = log_file_path
        self.save_images = save_images
        self.save_model_interval = save_model_interval
        self.save_model_path = save_model_path
