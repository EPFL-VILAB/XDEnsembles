'''
  Name: train_sig.py
  Desc: Executes training of a network with the consistency framework.
    Here are some options that may be specified for any model. If they have a
    default value, it is given at the end of the description in parens.
        Data pipeline:
            Data locations:
                'train_buildings': A list of the folders containing the training data. This
                    is defined in configs/split.txt.
                'val_buildings': As above, but for validation data.
                'data_dirs': The folder that all the data is stored in. This may just be
                    something like '/', and then all filenames in 'train_filenames' will
                    give paths relative to 'dataset_dir'. For example, if 'dataset_dir'='/',
                    then train_filenames might have entries like 'path/to/data/img_01.png'.
                    This is defiled in utils.py.
        Logging:
            'results_dir': An absolute path to where checkpoints are saved. This is
                defined in utils.py.
        Training:
            'batch_size': The size of each batch. (32)
            'max_epochs': The maximum number of epochs to train for. (800)
            'energy_config': {multiperceptual_targettask} The paths taken to compute the losses.
        Optimization:
            'learning_rate': The initial learning rate to use for the model. (3e-5)
  Usage:
    python -m train_sig trainsig_edgereshade --batch-size 32 --max-epochs 1000
'''

import torch
import torch.nn as nn

from utils import *
from energy_sig import get_energy_loss
from graph import TaskGraph
from logger import Logger, VisdomLogger
from datasets import load_train_val_sig, load_test, load_ood
from task_configs import tasks, RealityTask
from transfers import functional_transfers

from fire import Fire

import wandb
wandb.init(project="xdomain-ensembles", entity="robust_team")

def main(
	loss_config="trainsig_edgereshade", mode="standard", visualize=False,
	fast=False, batch_size=32, learning_rate=3e-5, resume=False,
	subset_size=None, max_epochs=800, dataaug=False, **kwargs,
):

	# CONFIG
	wandb.config.update({"loss_config":loss_config,"batch_size":batch_size,"lr":learning_rate})

	batch_size = batch_size or (4 if fast else 64)
	energy_loss = get_energy_loss(config=loss_config, mode=mode, **kwargs)

	# DATA LOADING
	train_undist_dataset, train_dist_dataset, val_ooddist_dataset, val_dist_dataset, val_dataset, train_step, val_step = load_train_val_sig(
		energy_loss.get_tasks("val"),
		batch_size=batch_size, fast=fast,
		subset_size=subset_size,
	)
	test_set = load_test(energy_loss.get_tasks("test"))

	ood_set = load_ood(energy_loss.get_tasks("ood"), ood_path='./assets/ood_natural/')
	ood_syn_aug_set = load_ood(energy_loss.get_tasks("ood_syn_aug"), ood_path='./assets/st_syn_distortions/')
	ood_syn_set = load_ood(energy_loss.get_tasks("ood_syn"), ood_path='./assets/ood_syn_distortions/', sample=35)

	train_undist = RealityTask("train_undist", train_undist_dataset, batch_size=batch_size, shuffle=True)
	train_dist = RealityTask("train_dist", train_dist_dataset, batch_size=batch_size, shuffle=True)
	val_ooddist = RealityTask("val_ooddist", val_ooddist_dataset, batch_size=batch_size, shuffle=True)
	val_dist = RealityTask("val_dist", val_dist_dataset, batch_size=batch_size, shuffle=True)
	val = RealityTask("val", val_dataset, batch_size=batch_size, shuffle=True)
	test = RealityTask.from_static("test", test_set, energy_loss.get_tasks("test"))

	ood = RealityTask.from_static("ood", ood_set, [tasks.rgb,])                                  ## standard ood set - natural
	ood_syn_aug = RealityTask.from_static("ood_syn_aug", ood_syn_aug_set, [tasks.rgb,])          ## synthetic distortion images used for sig training 
	ood_syn = RealityTask.from_static("ood_syn", ood_syn_set, [tasks.rgb,])                      ## unseen syn distortions

	# GRAPH
	realities = [train_undist, train_dist, val_ooddist, val_dist, val, test, ood, ood_syn_aug, ood_syn]
	graph = TaskGraph(tasks=energy_loss.tasks + realities, pretrained=True, finetuned=False,
		freeze_list=energy_loss.freeze_list,
	)
	graph.compile(torch.optim.Adam, lr=3e-5, weight_decay=2e-6, amsgrad=True)

	if resume:
		graph.load_weights('/workspace/shared/results_test_1/graph.pth')
		graph.optimizer.load_state_dict(torch.load('/workspace/shared/results_test_1/opt.pth'))
	# else:
	# 	folder_name='/workspace/shared/results_wavelet2normal_depthreshadecurvimgnetl1perceps_0.1nll/'
	# 	# pdb.set_trace()
	# 	in_domain='wav'
	# 	out_domain='normal'
	# 	graph.load_weights(folder_name+'graph.pth', [str((in_domain, out_domain))])
	# 	create_t0_graph(folder_name,in_domain,out_domain)
	# 	graph.load_weights(folder_name+'graph_t0.pth', [str((in_domain, f'{out_domain}_t0'))])


	# LOGGING
	logger = VisdomLogger("train", env=JOB)    # fake visdom logger
	logger.add_hook(lambda logger, data: logger.step(), feature="loss", freq=20)
	energy_loss.logger_hooks(logger)

	# BASELINE 
	if not resume:
		graph.eval()
		with torch.no_grad():
			for reality in [val_ooddist,val_dist,val]:
				for _ in range(0, val_step):
					val_loss = energy_loss(graph, realities=[reality])
					val_loss = sum([val_loss[loss_name] for loss_name in val_loss])
					reality.step()
					logger.update("loss", val_loss)
			for reality in [train_undist,train_dist]:
				for _ in range(0, train_step):
					train_loss = energy_loss(graph, realities=[reality])
					train_loss = sum([train_loss[loss_name] for loss_name in train_loss])
					reality.step()
					logger.update("loss", train_loss)
		
		energy_loss.logger_update(logger)
		data=logger.step()
		del data['loss']
		data = {k:v[0] for k,v in data.items()}
		wandb.log(data, step=0)

		path_values = energy_loss.plot_paths(graph, logger, realities, prefix="")
		for reality_paths, reality_images in path_values.items():
			wandb.log({reality_paths: [wandb.Image(reality_images)]}, step=0)


	# TRAINING
	for epochs in range(0, max_epochs):

		logger.update("epoch", epochs)

		graph.train()
		for _ in range(0, train_step):
			train_loss_nll = energy_loss(graph, realities=[train_undist])
			train_loss_nll = sum([train_loss_nll[loss_name] for loss_name in train_loss_nll])
			train_loss_lwfsig = energy_loss(graph, realities=[train_dist])
			train_loss_lwfsig = sum([train_loss_lwfsig[loss_name] for loss_name in train_loss_lwfsig])
			train_loss = train_loss_nll+train_loss_lwfsig
			graph.step(train_loss)
			train_undist.step()
			train_dist.step()
			logger.update("loss", train_loss)

		graph.eval()
		for _ in range(0, val_step):
			with torch.no_grad():
				val_loss = energy_loss(graph, realities=[val_dist])
				val_loss = sum([val_loss[loss_name] for loss_name in val_loss])
			val_dist.step()
			logger.update("loss", val_loss)
		
		if epochs % 20 == 0:
			for reality in [val,val_ooddist]:
				for _ in range(0, val_step):
					with torch.no_grad():
						val_loss = energy_loss(graph, realities=[reality])
						val_loss = sum([val_loss[loss_name] for loss_name in val_loss])
					reality.step()
					logger.update("loss", val_loss)

		energy_loss.logger_update(logger)

		data=logger.step()
		del data['loss']
		data = {k:v[0] for k,v in data.items()}
		wandb.log(data, step=epochs+1)

		if epochs % 10 == 0:
			graph.save(f"{RESULTS_DIR}/graph.pth")
			torch.save(graph.optimizer.state_dict(),f"{RESULTS_DIR}/opt.pth")

		if (epochs % 100 == 0) or (epochs % 15 == 0 and epochs <= 30):
			path_values = energy_loss.plot_paths(graph, logger, realities, prefix="")
			for reality_paths, reality_images in path_values.items():
				wandb.log({reality_paths: [wandb.Image(reality_images)]}, step=epochs+1)



	graph.save(f"{RESULTS_DIR}/graph.pth")
	torch.save(graph.optimizer.state_dict(),f"{RESULTS_DIR}/opt.pth")

if __name__ == "__main__":
	Fire(main)
