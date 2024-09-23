
import argparse
import random
from van_code.runners import *



parser = argparse.ArgumentParser()
#### Theory
parser.add_argument("--Lc", type=int, default=2)
parser.add_argument("--beta", type=float, default=0.44)


##### PixelCNN
parser.add_argument("--net_depth", type=int, default=3)
parser.add_argument("--net_width", type=int, default=16)
parser.add_argument("--half_kernel_size", type=int, default=6)
parser.add_argument("--bias", action="store_true")
parser.add_argument("--not_z2", action="store_false")
parser.add_argument("--not_residual", action="store_false")
parser.add_argument("--x_hat_clip", action="store_true")
parser.add_argument("--eps", type=float, default=1.e-7)


####### Course/fine net
parser.add_argument("--hidden_sizes", nargs="*", type=int, default=[16])
parser.add_argument("--kernel_size", nargs="*", type=int, default=[5,3])

### Multilevels

parser.add_argument("--n_blocks", type=int, default=3)
parser.add_argument("--not_hb_last", action="store_false")

### training
parser.add_argument("--train", action="store_true")
parser.add_argument("--epochs", nargs="*", type=int, default=[1000,1500,1500,1500,1500,1500,1500])
parser.add_argument("--bs", nargs="*", type=int,default=[10000,10000,10000,500,100,96,16]) #default=[1024,1024,1024,512,256,96,16])
parser.add_argument("--lr",type=float, default=0.001 )
parser.add_argument("--print_freq", type=int, default=100)
parser.add_argument("--patience", type=int, default=1000)
#######
#######
#WARNING!!!!!!
#If you train with vanilla (standard) method be careful because weights are
#saved in the same dir of others trainings
#Use the argumnet --hyp with an integer number to generate different directories
#for weights and metrics
#!!!!!!!!
#########
parser.add_argument("--vanilla_train", action="store_true") #vannilla train
parser.add_argument("--vanilla_epochs", type=int, default=100)
parser.add_argument("--vanilla_bs", type=int, default=16)
#### Sampling
parser.add_argument("--measures", action="store_true")
parser.add_argument("--measures_md", action="store_true") #mode dropping
parser.add_argument("--measures_imh", action="store_true") #mode independent MH
parser.add_argument("--bs_eval", type=int, default=1000)
parser.add_argument("--nmeas", type=int, default=1000)
parser.add_argument("--data_cluster", type=int, default=1000000)

## Paths
parser.add_argument("--main_path", type=str, default="/leonardo_work/INF24_sft_1/ecellini/multilevelRG/" )
###Use the following to generate different directories for same model in order
#to not overwrite files
parser.add_argument("--hyp", type=int, default=0 )
parser.add_argument("--fix_seed", action="store_true")
parser.add_argument("--seed", type=int, default=137)

args = parser.parse_args()

if args.fix_seed:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)


def main(args):
    if args.train:
        training(args)

    if args.measures:
        measures(args)

    if args.measures_md:
        measures_modedrop(args)

    if args.measures_imh:
        measures_IMH(args)


def get_args():
    return parser.parse_args()


if __name__ == "__main__":
    main(args)
