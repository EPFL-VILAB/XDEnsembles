# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch
import torch.nn as nn

from utils import *
from energy_merging import get_energy_loss
from graph import TaskGraph
from logger import Logger, VisdomLogger
from datasets import load_train_val_merging, load_test, load_ood
from task_configs import tasks, RealityTask
from transfers import functional_transfers

from fire import Fire

import wandb
wandb.init(project="xdomain-ensembles", entity="robust_team")


def main(
	loss_config="multiperceptual", mode="standard", visualize=False,
	fast=False, batch_size=None,
	subset_size=None, max_epochs=5000, dataaug=False, **kwargs,
):

	# CONFIG
	wandb.config.update({"loss_config":loss_config,"batch_size":batch_size,"data_aug":dataaug,"lr":"3e-5",
		"n_gauss":1,"distribution":"laplace"})

	batch_size = batch_size or (4 if fast else 64)
	energy_loss = get_energy_loss(config=loss_config, mode=mode, **kwargs)

	# DATA LOADING
	train_dataset, val_dataset, val_noaug_dataset, train_step, val_step = load_train_val_merging(
		energy_loss.get_tasks("train_c"),
		batch_size=batch_size, fast=fast,
		subset_size=subset_size,
	)
	test_set = load_test(energy_loss.get_tasks("test"))

	ood_set = load_ood(energy_loss.get_tasks("ood"), ood_path='./assets/ood_natural/')
	ood_syn_aug_set = load_ood(energy_loss.get_tasks("ood_syn_aug"), ood_path='./assets/st_syn_distortions/')
	ood_syn_set = load_ood(energy_loss.get_tasks("ood_syn"), ood_path='./assets/ood_syn_distortions/', sample=35)

	train = RealityTask("train_c", train_dataset, batch_size=batch_size, shuffle=True)      # distorted and undistorted 
	val = RealityTask("val_c", val_dataset, batch_size=batch_size, shuffle=True)            # distorted and undistorted 
	val_noaug = RealityTask("val", val_noaug_dataset, batch_size=batch_size, shuffle=True)  # no augmentation
	test = RealityTask.from_static("test", test_set, energy_loss.get_tasks("test"))

	ood = RealityTask.from_static("ood", ood_set, [tasks.rgb,])                                  ## standard ood set - natural
	ood_syn_aug = RealityTask.from_static("ood_syn_aug", ood_syn_aug_set, [tasks.rgb,])          ## synthetic distortion images used for sig training 
	ood_syn = RealityTask.from_static("ood_syn", ood_syn_set, [tasks.rgb,])                      ## unseen syn distortions

	# GRAPH
	realities = [train, val, val_noaug, test, ood, ood_syn_aug, ood_syn]
	graph = TaskGraph(tasks=energy_loss.tasks + realities, pretrained=True, finetuned=False,
		freeze_list=energy_loss.freeze_list,
	)
	graph.compile(torch.optim.Adam, lr=3e-5, weight_decay=2e-6, amsgrad=True)

	# LOGGING
	logger = VisdomLogger("train", env=JOB)    # fake visdom logger
	logger.add_hook(lambda logger, data: logger.step(), feature="loss", freq=20)
	energy_loss.logger_hooks(logger)

	graph.eval()
	path_values = energy_loss.plot_paths(graph, logger, realities, prefix="")
	for reality_paths, reality_images in path_values.items():
		wandb.log({reality_paths: [wandb.Image(reality_images)]}, step=0)

	with torch.no_grad():
		for reality in [val,val_noaug]:
			for _ in range(0, val_step):
				val_loss = energy_loss(graph, realities=[reality])
				val_loss = sum([val_loss[loss_name] for loss_name in val_loss])
				reality.step()
				logger.update("loss", val_loss)

		for _ in range(0, train_step):
			train_loss = energy_loss(graph, realities=[train], compute_grad_ratio=True)
			train_loss = sum([train_loss[loss_name] for loss_name in train_loss])
			train.step()
			logger.update("loss", train_loss)

	energy_loss.logger_update(logger)

	data=logger.step()
	del data['loss']
	data = {k:v[0] for k,v in data.items()}
	wandb.log(data, step=0)

	# TRAINING
	for epochs in range(0, max_epochs):

		logger.update("epoch", epochs)

		graph.train()
		for _ in range(0, train_step):
			train_loss = energy_loss(graph, realities=[train], compute_grad_ratio=True)
			train_loss = sum([train_loss[loss_name] for loss_name in train_loss])
			graph.step(train_loss)
			train.step()
			logger.update("loss", train_loss)

		graph.eval()
		for reality in [val,val_noaug]:
			for _ in range(0, val_step):
				with torch.no_grad():
					val_loss = energy_loss(graph, realities=[reality])
					val_loss = sum([val_loss[loss_name] for loss_name in val_loss])
				reality.step()
				logger.update("loss", val_loss)

		energy_loss.logger_update(logger)

		data=logger.step()
		del data['loss']
		del data['epoch']
		data = {k:v[0] for k,v in data.items()}
		wandb.log(data, step=epochs+1)

		if epochs % 10 == 0:
			graph.save(f"{RESULTS_DIR}/graph.pth")
			torch.save(graph.optimizer.state_dict(),f"{RESULTS_DIR}/opt.pth")

		if epochs % 25 == 0:
			path_values = energy_loss.plot_paths(graph, logger, realities, prefix="")
			for reality_paths, reality_images in path_values.items():
				wandb.log({reality_paths: [wandb.Image(reality_images)]}, step=epochs+1)



	graph.save(f"{RESULTS_DIR}/graph.pth")
	torch.save(graph.optimizer.state_dict(),f"{RESULTS_DIR}/opt.pth")

if __name__ == "__main__":
	Fire(main)
