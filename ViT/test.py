import argparse
import os
import torch
import torch.backends.cudnn as cudnn

import utils
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import models_vit_mae
from eval import evaluate

import saint

def get_args_parser():
    parser = argparse.ArgumentParser('MAE Token Pruning (Adapted from MAE repo)', add_help=False)
    parser.add_argument('--batch_size', default=256, type=int)
    # Model parameters
    parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets/imagenet/', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    
    
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--ckpt_path', default='ckpts/mae_finetuned_vit_base.pth',
                        help='checkpoint')
    parser.add_argument('--benchmark_throughput', action='store_true',
                        help='Perform throughput benchmarking')
    
    parser.add_argument('--benchmark_gflops', action='store_true',
                        help='Perform gflops benchmarking')
    
    ### SAINT
    parser.add_argument('--sim_threshold', default=0.7, type=float,
                        help='Sim Threshold in SAINT alg.')
    parser.add_argument('--K', default=5, type=int,
                        help='K in SAINT alg.')
    parser.add_argument('--gamma', default=10.0, type=float,
                        help='Gamma in SAINT alg.')
    

    return parser


def main(args):
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)
    cudnn.benchmark = True
    
    transform_val = transforms.Compose([
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    dataset_val = datasets.ImageFolder(os.path.join(args.data_path, 'val'), transform=transform_val)
    print(dataset_val)

    sampler_val = torch.utils.data.RandomSampler(dataset_val)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )
    
    
    ## 0: loading model from checkpoint 
    model = models_vit_mae.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_path_rate=0.1,
        global_pool=True,
    )

    utils.load_model(path=args.ckpt_path, model=model) 


    ### ViTs
    # model_name = 'vit_small_patch16_224'
    # model_name = 'vit_base_patch16_224'
    # model_name = 'vit_large_patch16_224'
    # model_name = 'vit_huge_patch14_224'

    
    
    # model_name = 'deit_small_patch16_224'
    # model_name = 'deit_base_patch16_224'
    # model_name = 'deit3_huge_patch14_224'
    
    
    # model_name = 'deit3_small_patch16_384'
    # model_name = 'deit3_base_patch16_384'
    # model_name = 'deit3_large_patch16_384'
    

    # model_name = 'vit_large_patch14_clip_336'

    # model = timm.create_model(model_name, pretrained=True)

    
    
    # saint.patch.timm(model)
    
    # model.prune_mode = 'drop'
    
    
    # saint.patch.mae(model)
    # count = len(model.blocks) // 2
    # model.prune_mode = ['drop'] * count
    # model.sim_threshold = [0.8] *  count
    
    # saint.patch.mae(model)
    # count = len(model.blocks) // 2
    # model.prune_mode = ['drop'] * count
    # model.sim_threshold = [0.75] *  count
    
    
    
    
    
    model.to(device)
    
    #### SAINT
    saint.patch.mae(model, args.sim_threshold, args.K, args.gamma)
    # saint.patch.timm(model)

    count = len(model.blocks) // 2
    model.prune_mode = ['drop'] * count
    model.sim_threshold = [args.sim_threshold] * count # one could have different thresholds per each layer
    



    benchmark_macs = False
    if benchmark_macs:
        from torchprofile import profile_macs
        warm_up = 0
        runs = 2
        total = 0
        total_macs = 0
        batch_size = args.batch_size
        it = iter(data_loader_val)
        with torch.no_grad():
            for i in tqdm(range(runs), disable=False, desc="Benchmarking"):
                input, _ = next(it)
                input = input.to('cuda')
                
                macs = profile_macs(model, input)
                print(f"macs:{macs/1e9}GFLOPs")
                total_macs += macs/1e9
                total += batch_size

        gflops = total_macs / total
        
        print(f"gflops: {gflops}")
    

    
    # exit()
    
    args.benchmark = False
    if args.benchmark:
        
        warm_up = 15
        runs = 35
        total = 0
        batch_size = args.batch_size
        it = iter(data_loader_val)
        with torch.no_grad():
            for i in tqdm(range(runs), disable=False, desc="Benchmarking"):
                if i == warm_up:
                    torch.cuda.synchronize()
                    total = 0
                    start = time.time()

                input, _ = next(it)
                input = input.to('cuda')
                
                model(input)
                total += batch_size


        
        torch.cuda.synchronize()

        end = time.time()
        elapsed = end - start

        throughput = total / elapsed
        
        print(f"Throughput: {throughput}")
    
    else:
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")

    
    


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)

    