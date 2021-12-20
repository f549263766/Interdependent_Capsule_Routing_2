"""
@author: QYZ
@time: 2021/11/26
@file: model_select.py
@describe: This file aims to select model.
"""


def model_selected(parser, device):
    args = parser.parse_args()
    if args.model == "VB_Routing":
        parser.add_argument("--pose_dim", type=int, default=4,
                            help="Dimensions of the gestalt matrix")
        parser.add_argument("--routing_iter", type=int, default=3,
                            help="Number of routing algorithm iterations")
        parser.add_argument("--arch", nargs='+', type=int, default=[64, 16, 32, 32, 5],
                            help="Number of output channels per capsule layer")
        args = parser.parse_args()
        from models.vb_capsnet import CapsuleNet
        model = CapsuleNet(args)
        return model.to(device), args

    elif args.model == "CLA_Routing":
        parser.add_argument("--pose_dim", type=int, default=4,
                            help="Dimensions of the gestalt matrix")
        parser.add_argument("--arch", nargs='+', type=int, default=[32, 8, 8, 8, 5],
                            help="Number of output channels per capsule layer")
        parser.add_argument("--feature", type=bool, default=True,
                            help="Whether to use feature")
        parser.add_argument("--feature_dim", type=int, default=16,
                            help="Number of classes")
        args = parser.parse_args()
        from models.cla_capsnet import CLACapsNet
        model = CLACapsNet(args)
        return model.to(device), args
