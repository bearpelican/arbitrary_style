#!/usr/bin/env python

import argparse
import ncluster
import os

IMAGE_NAME = 'Deep Learning AMI (Ubuntu) Version 16.0'
INSTANCE_TYPE = 'c5.2xlarge'

ncluster.set_backend('aws')
parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='style_compute',
                    help="name of the current run, used for machine naming and tensorboard visualization")
parser.add_argument('--machines', type=int, default=1,
                    help="how many machines to use")
args = parser.parse_args()

def main():
  supported_regions = ['us-west-2', 'us-east-1', 'us-east-2']
  assert ncluster.get_region() in supported_regions, f"required AMI {IMAGE_NAME} has only been made available in regions {supported_regions}, but your current region is {ncluster.get_region()}"

#   os.environ['NCLUSTER_AWS_FAST_ROOTDISK'] = '1'  # use io2 disk on AWS
  job = ncluster.make_job(name=args.name,
                          run_name=f"{args.name}-{args.machines}",
                          num_tasks=args.machines,
                          image_name=IMAGE_NAME,
                          instance_type=INSTANCE_TYPE,
                          disk_size=300,
                        #   install_script=open('setup.sh').read(),
                        #   skip_efs=False,
                        #   spot=True
                          )
  job.upload('training')
  job.run(f'source activate fastai')
  job.run(f'cd training')
  job.run(f'jnb')



if __name__ == '__main__':
  main()
