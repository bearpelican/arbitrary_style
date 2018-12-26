#!/usr/bin/env python

import argparse
import ncluster
import os

IMAGE_NAME = 'style_transfer_v0'
INSTANCE_TYPE = 'p3.16xlarge'
NUM_GPUS = {'p3.2xlarge': 1, 'p3.8xlarge':4, 'p3.16xlarge':8}[INSTANCE_TYPE]

ncluster.set_backend('aws')
parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='style',
                    help="name of the current run, used for machine naming and tensorboard visualization")
parser.add_argument('--machines', type=int, default=1,
                    help="how many machines to use")
parser.add_argument('--fastai', action='store_true', help='Run fastai script.')
args = parser.parse_args()

lr = 1.0
bs = [512, 224, 128] # largest batch size that fits in memory for each image size
bs_scale = [x/bs[0] for x in bs]
one_machine = [
  {'ep':(0,1),  'lr':(lr,lr*2)}, # lr warmup is better with --init-bn0
  {'ep':(1,2), 'lr':(lr*2,lr/4)}, # trying one cycle
]

schedules = {1: one_machine}


# routines to build NCCL ring orders
def get_nccl_params(num_tasks, num_gpus):
  if num_tasks <= 1:
    return 'NCCL_DEBUG=VERSION'
  # return 'NCCL_MIN_NRINGS=2 NCCL_SINGLE_RING_THRESHOLD=10 NCCL_DEBUG=VERSION'

def format_params(arg):
  if isinstance(arg, list) or isinstance(arg, dict):
    return '\"' + str(arg) + '\"'
  else:
    return str(arg)


def main():
  supported_regions = ['us-west-2', 'us-east-1', 'us-east-2']
  assert ncluster.get_region() in supported_regions, f"required AMI {IMAGE_NAME} has only been made available in regions {supported_regions}, but your current region is {ncluster.get_region()}"
  assert args.machines in schedules, f"{args.machines} not supported, only support {schedules.keys()}"

  os.environ['NCLUSTER_AWS_FAST_ROOTDISK'] = '1'  # use io2 disk on AWS
  job = ncluster.make_job(name=args.name,
                          run_name=f"{args.name}-{args.machines}",
                          num_tasks=args.machines,
                          image_name=IMAGE_NAME,
                          instance_type=INSTANCE_TYPE,
                        #   disk_size=1000,
                        #   install_script=open('setup.sh').read(),
                        #   skip_efs=False,
                          spot=True
                          )
  job.upload('training')
  job.run(f'conda activate fastai')

  nccl_params = get_nccl_params(args.machines, NUM_GPUS)

  # Training script args
  default_params = [
      # '--load', f'/ncluster/models/{args.name}.pth',
      '--load', f'/ncluster/models/models/fastai_style_retrain_v3_6.pth',
      '--dist-url', 'file:///home/ubuntu/data/file.sync',
      '--resnet',
      '--save', f'/ncluster/models/{args.name}.pth'
      ]

  params = ['--phases', schedules[args.machines]]
  training_params = default_params + params
  training_params = ' '.join(map(format_params, training_params))
  train_script = 'training/train_fastai.py' if args.fastai else 'training/train.py'

  # TODO: simplify args processing, or give link to actual commands run
  for i, task in enumerate(job.tasks):
    
    dist_params = f'--nproc_per_node={NUM_GPUS} --nnodes={args.machines} --node_rank={i} --master_addr={job.tasks[0].ip} --master_port={6006}'
    cmd = f'{nccl_params} python -m torch.distributed.launch {dist_params} {train_script} {training_params}'
    # task.run(f'echo {cmd} > {job.logdir}/task-{i}.cmd')  # save command-line
    task.run(cmd, non_blocking=True)

#   print(f"Logging to {job.logdir}")


if __name__ == '__main__':
  main()
