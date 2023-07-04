from avalanche.benchmarks.classic import CORe50
from torch.utils.data.dataset import ConcatDataset
import torch
from avalanche.evaluation.metrics import forgetting_metrics, class_accuracy_metrics, \
accuracy_metrics, loss_metrics, timing_metrics, cpu_usage_metrics, \
confusion_matrix_metrics, disk_usage_metrics, gpu_usage_metrics, ram_usage_metrics, \
forward_transfer_metrics
from avalanche.training.plugins import EvaluationPlugin, EWCPlugin, EarlyStoppingPlugin
from avalanche.training.templates import SupervisedTemplate
from avalanche.logging import InteractiveLogger, TextLogger, WandBLogger, TensorboardLogger
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from avalanche.models import SimpleCNN, MlpVAE, SimpleMLP, SlimResNet18, PNN, MLPAdapter
from avalanche.training.supervised import Naive, Cumulative, EWC, GenerativeReplay, VAETraining, Replay, GEM, PNNStrategy, CWRStar
from avalanche.training.plugins import GenerativeReplayPlugin, CWRStarPlugin, GEMPlugin
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import pandas as pd 
import wandb
from torch.utils.data import ConcatDataset
from avalanche.benchmarks.utils.data_loader import TaskBalancedDataLoader

from avalanche.benchmarks.utils import make_classification_dataset
from sklearn.model_selection import train_test_split
from avalanche.benchmarks.generators import benchmark_with_validation_stream, class_balanced_split_strategy


PATIENCE = 15

def main(n_strategies):
  
    # Load the CORe50 dataset
    core50 = CORe50(scenario="nc", mini=True, object_lvl=True)

    # the task label of each train_experience.
    print('--- Task labels:')
    print(core50.task_labels)

    # Instatiate train and test stream
    train_stream = core50.train_stream
    test_stream = core50.test_stream

    if torch.cuda.is_available():
        device = 'cuda:0'
        torch.backends.cudnn.benchmark = True
        device_count = torch.cuda.device_count()
        print(f"Found {device_count} CUDA GPU devices.")
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    print(f'Using {device} device')

    # device = 'cpu'

    loggers = []
    # # log to Tensorboard
    # loggers.append(TensorboardLogger())
    # # log to text file
    # loggers.append(TextLogger(open('log.txt', 'a')))
    # # print to stdout
    # loggers.append(InteractiveLogger())

    for i in range(n_strategies):
        run_name = f"trial{i}"
        print(run_name)
        loggers.append(WandBLogger(project_name="avalanche", run_name=run_name, params={"reinit": True}))
    print(len(loggers))

    # DEFINE THE EVALUATION PLUGIN
    # The evaluation plugin manages the metrics computation.
    # It takes as argument a list of metrics, collectes their results and returns
    # them to the strategy it is attached to.
    eval_plugin = []
    for i in range(n_strategies):
        eval_plugin.append(EvaluationPlugin(
            accuracy_metrics(stream=True),
            cpu_usage_metrics(stream=True),
            gpu_usage_metrics(gpu_id=0, stream=True),
            ram_usage_metrics(stream=True),
            disk_usage_metrics(stream=True),
            loggers=loggers[i],
            strict_checks=False
        ))

    # Define the models
    model = SimpleCNN(num_classes=50).to(device)
    # model = SimpleMLP(num_classes=50, input_size=32*32*3, hidden_size=32*32*3, hidden_layers=4, drop_rate=0.5).to(device)
    # pnn_model = PNN(num_layers=4, in_features=32*32*3, hidden_features_per_column=32*32*3).to(device)
    # gr_model = MlpVAE((3, 32, 32), nhid=2, n_classes=50, device=device)
    print(f"Main model {model}")
    # print(f"PNN model {pnn_model}")
    # print(f"GR model {gr_model}")
    # model = MlpVAE((3, 32, 32), nhid=2, device=device)

    # model = MTSimpleCNN().to(device)

    # model = SimpleMLP(num_classes=50, input_size=(3,32,32), hidden_size=32, hidden_layers = 2, drop_rate=0.5)

    # model = SlimResNet18(nclasses=50)

    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = CrossEntropyLoss()
    epochs = 20
    batchsize_train = 100
    batchsize_eval = 100

    cl_strategies = [
    CWRStar(
        model, optimizer, criterion, cwr_layer_name=None, device=device,
        train_mb_size=batchsize_train, train_epochs=epochs, eval_mb_size=batchsize_eval, evaluator=eval_plugin[0]
     ),     
    # VAETraining(
    #     gr_model, optimizer, device=device,
    #     train_mb_size=batchsize_train, train_epochs=epochs, eval_mb_size=batchsize_eval, evaluator=eval_plugin,
    #     plugins=[GenerativeReplayPlugin()],
    # ),        
    # PNNStrategy(
    #     pnn_model, optimizer, criterion, device=device,
    #     train_mb_size=batchsize_train, train_epochs=epochs, eval_mb_size=batchsize_eval, evaluator=eval_plugin
    #  ), 
    GEM(
        model, optimizer, criterion, device=device,
        train_mb_size=batchsize_train, train_epochs=epochs, eval_mb_size=batchsize_eval, evaluator=eval_plugin[1],
        patterns_per_exp=5
     ), 
    Cumulative(
        model, optimizer, criterion, device=device,
        train_mb_size=batchsize_train, train_epochs=epochs, eval_mb_size=batchsize_eval, evaluator=eval_plugin[2]
     ), 
    EWC(
        model, optimizer, criterion, ewc_lambda=1.0, device=device,
        train_mb_size=batchsize_train, train_epochs=epochs, eval_mb_size=batchsize_eval, evaluator=eval_plugin[3]
    ), 
    Naive(
        model, optimizer, criterion, device=device,
        train_mb_size=batchsize_train, train_epochs=epochs, eval_mb_size=batchsize_eval, evaluator=eval_plugin[4]
    )]

    cl_strategies = []
    # TRAINING LOOP
    results = []
    for cl_strategy in cl_strategies:
        print(f"Current training strategy: {cl_strategy}")
        for train_experience in train_stream:
            print(f"Experience number {train_experience.current_experience}")
            print(f"Classes seen so far {train_experience.classes_seen_so_far}")
            print(f"Training on {len(train_experience.dataset)} examples")
   
            cl_strategy.train(train_experience)

        # Evaluate on test set
        print(f"Testing on {len(test_stream[0].dataset)} examples")
        results.append(cl_strategy.eval(test_stream))
        # print(f"Results so far {results}")

def GR_Plugin():

    if torch.cuda.is_available():
        device = 'cuda:0'
        torch.backends.cudnn.benchmark = True
        device_count = torch.cuda.device_count()
        print(f"Found {device_count} CUDA GPU devices.")
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    print(f'Using {device} device')


    # --- BENCHMARK CREATION
    benchmark = CORe50(scenario="nc", mini=True, object_lvl=True)
    # ---------

    # MODEL CREATION
    model = MlpVAE((3, 32, 32), nhid=2, device=device)

    # CREATE THE STRATEGY INSTANCE (GenerativeReplay)
    cl_strategy = VAETraining(
        model,
        torch.optim.Adam(model.parameters(), lr=0.001),
        train_mb_size=100,
        train_epochs=4,
        device=device,
        plugins=[GenerativeReplayPlugin()],
    )

    # TRAINING LOOP
    print("Starting experiment...")
    f, axarr = plt.subplots(benchmark.n_experiences, 10)
    k = 0
    for train_experience in benchmark.train_stream:
        print("Start of train_experience ", train_experience.current_experience)
        cl_strategy.train(train_experience)
        print("Training completed")

        samples = model.generate(10)
        samples = samples.detach().cpu().numpy()

        for j in range(10):
            axarr[k, j].imshow(samples[j, 0], cmap="gray")
            axarr[k, 4].set_title("Generated images for train_experience " + str(k))
        np.vectorize(lambda ax: ax.axis("off"))(axarr)
        k += 1

    f.subplots_adjust(hspace=1.2)
    plt.savefig("VAE_output_per_exp")
    plt.show()

def cumulative_only(group_name, num_runs):

    if torch.cuda.is_available():
        device = 'cuda:0'
        torch.backends.cudnn.benchmark = True
        device_count = torch.cuda.device_count()
        print(f"Found {device_count} CUDA GPU devices.")
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    print(f'Using {device} device')

    model = SimpleCNN(num_classes=50).to(device)
    print(f"Main model {model}")

    # Load the CORe50 dataset
    core50 = CORe50(scenario="nc", mini=True, object_lvl=True)

    # the task label of each train_experience.
    print('--- Task labels:')
    print(core50.task_labels)

    # Instatiate train, validation and test streams
    core50 = benchmark_with_validation_stream(core50, 0.2)
    train_stream = core50.train_stream
    val_stream = core50.valid_stream
    test_stream = core50.test_stream


    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = CrossEntropyLoss()
    epochs = 10
    batchsize_train = 100
    batchsize_eval = 100
    patience = PATIENCE + 20
    eval_every = 1



    for i in range(num_runs):

        loggers = []
        loggers.append(WandBLogger(project_name=project_name, run_name="Cumulative", params={"reinit": True, "group": group_name}))
        loggers.append(InteractiveLogger())
        loggers.append(TextLogger(open(f'Cumulative_{i}.txt', 'a')))

        eval_plugin = EvaluationPlugin(
        accuracy_metrics(stream=True),
        class_accuracy_metrics(stream=True),
        cpu_usage_metrics(epoch=True),
        gpu_usage_metrics(gpu_id=0, epoch=True),
        ram_usage_metrics(epoch=True),
        disk_usage_metrics(epoch=True),
        # forward_transfer_metrics(stream=True),
        forgetting_metrics(stream=True),
        loggers=loggers,
        strict_checks=True)


        cl_strategy = Cumulative(
            model, optimizer, criterion, device=device,
            train_mb_size=batchsize_train, train_epochs=epochs, eval_mb_size=batchsize_eval, evaluator=eval_plugin, \
            plugins=[EarlyStoppingPlugin(patience, "valid", "Top1_Acc_Stream")], eval_every=eval_every)

        print(f"Current training strategy: {cl_strategy}")
        for train_experience, val_experience in zip(train_stream, val_stream):
            print(f"Experience number train {train_experience.current_experience}")
            print(f"classes in this experience train {train_experience.classes_in_this_experience}")
            print(f"Classes seen so far train {train_experience.classes_seen_so_far}")
            print(f"Training on train {len(train_experience.dataset)} examples")
            print(f"Experience number validation {val_experience.current_experience}")
            print(f"Classes in this experience val {val_experience.classes_in_this_experience}")
            print(f"Classes seen so far validation {val_experience.classes_seen_so_far}")
            print(f"Validating on validation {len(val_experience.dataset)} examples")
            # cl_strategy.train_dataset_adaptation(train_experience)  

            cl_strategy.train(train_stream, eval_streams=[val_stream])

        cl_strategy.eval(test_stream)

    # Finish the current WandB run
    wandb.finish()


def cumulative_via_plugin(group_name, num_runs):

    if torch.cuda.is_available():
        device = 'cuda:0'
        torch.backends.cudnn.benchmark = True
        device_count = torch.cuda.device_count()
        print(f"Found {device_count} CUDA GPU devices.")
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    print(f'Using {device} device')

    model = SimpleCNN(num_classes=50).to(device)
    print(f"Main model {model}")

    # Load the CORe50 dataset
    core50 = CORe50(scenario="nc", mini=True, object_lvl=True)

    # the task label of each train_experience.
    print('--- Task labels:')
    print(core50.task_labels)

    # Instatiate train, validation and test streams
    # core50 = benchmark_with_validation_stream(core50, 0.2, shuffle=False)

    validation_size = 0.2
    foo = lambda exp: class_balanced_split_strategy(validation_size, exp)
    core50 = benchmark_with_validation_stream(core50, custom_split_strategy=foo)

    train_stream = core50.train_stream
    val_stream = core50.valid_stream
    test_stream = core50.test_stream


    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = CrossEntropyLoss()
    epochs = 10
    batchsize_train = 100
    batchsize_eval = 100
    patience = PATIENCE + 20
    eval_every = 1


    for i in range(num_runs):

        loggers = []
        loggers.append(WandBLogger(project_name=project_name, run_name="Cumulative", params={"reinit": True, "group": group_name}))
        loggers.append(InteractiveLogger())
        loggers.append(TextLogger(open(f'Cumulative_{i}.txt', 'a')))

        eval_plugin = EvaluationPlugin(
        accuracy_metrics(epoch=True, stream=True),
        class_accuracy_metrics(stream=True),
        cpu_usage_metrics(epoch=True),
        gpu_usage_metrics(gpu_id=0, epoch=True),
        ram_usage_metrics(epoch=True),
        disk_usage_metrics(epoch=True),
        # forward_transfer_metrics(stream=True),
        forgetting_metrics(stream=True),
        loggers=loggers,
        strict_checks=True)


        cl_strategy = Cumulative(
            model, optimizer, criterion, device=device,
            train_mb_size=batchsize_train, train_epochs=epochs, eval_mb_size=batchsize_eval, evaluator=eval_plugin, \
            plugins=[EarlyStoppingPlugin(patience, "valid", "Top1_Acc_Stream")], eval_every=eval_every)

        print(f"Current training strategy: {cl_strategy}")
        for train_experience, val_experience in zip(train_stream, val_stream):
            print(f"Experience number train {train_experience.current_experience}")
            print(f"classes in this experience train {train_experience.classes_in_this_experience}")
            print(f"Classes seen so far train {train_experience.classes_seen_so_far}")
            print(f"Training on train {len(train_experience.dataset)} examples")
            print(f"Experience number validation {val_experience.current_experience}")
            print(f"Classes in this experience val {val_experience.classes_in_this_experience}")
            print(f"Classes seen so far validation {val_experience.classes_seen_so_far}")
            print(f"Validating on validation {len(val_experience.dataset)} examples")

            cl_strategy.train(train_experience, eval_streams=[val_experience])

        cl_strategy.eval(test_stream)

    # Finish the current WandB run
    wandb.finish()

def train_with_ES(group_name, num_runs):

    if torch.cuda.is_available():
        device = 'cuda:0'
        torch.backends.cudnn.benchmark = True
        device_count = torch.cuda.device_count()
        print(f"Found {device_count} CUDA GPU devices.")
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    print(f'Using {device} device')

    object_lvl=False

    if object_lvl:
        num_classes=50
    else:
        num_classes=10

    model = SimpleCNN(num_classes=num_classes).to(device)
    print(f"Main model {model}")

    # Load the CORe50 dataset
    core50 = CORe50(scenario="ni", mini=True, object_lvl=object_lvl)

    # the task label of each train_experience.
    print('--- Task labels:')
    print(core50.task_labels)

    # Instatiate train, validation and test streams
    validation_size = 0.2
    foo = lambda exp: class_balanced_split_strategy(validation_size, exp)
    core50 = benchmark_with_validation_stream(core50, custom_split_strategy=foo)

    train_stream = core50.train_stream
    val_stream = core50.valid_stream
    test_stream = core50.test_stream


    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = CrossEntropyLoss()
    epochs = 150
    batchsize_train = 100
    batchsize_eval = 100
    patience = PATIENCE
    eval_every = 1


    for i in range(num_runs):

        loggers = []
        loggers.append(WandBLogger(project_name=project_name, run_name="Naive", params={"reinit": True, "group": group_name}))
        loggers.append(InteractiveLogger())
        loggers.append(TextLogger(open(f'Naive_{i}.txt', 'a')))

        eval_plugin = EvaluationPlugin(
        accuracy_metrics(epoch=True, stream=True, trained_experience=True),
        class_accuracy_metrics(epoch=True, stream=True),
        cpu_usage_metrics(epoch=True),
        gpu_usage_metrics(gpu_id=0, epoch=True),
        ram_usage_metrics(epoch=True),
        disk_usage_metrics(epoch=True),
        # forward_transfer_metrics(stream=True),
        forgetting_metrics(stream=True),
        loggers=loggers,
        strict_checks=False)

        cl_strategy = Naive(
            model, optimizer, criterion, device=device,
            train_mb_size=batchsize_train, train_epochs=epochs, eval_mb_size=batchsize_eval, evaluator=eval_plugin, \
            eval_every=eval_every, plugins=[EarlyStoppingPlugin(patience, "valid", "Top1_Acc_Stream")])
        

        print(f"Current training strategy: {cl_strategy}")
        for train_experience, val_experience in zip(train_stream, val_stream):

            print(f"Experience number train {train_experience.current_experience}")
            print(f"classes in this experience train {train_experience.classes_in_this_experience}")
            print(f"Classes seen so far train {train_experience.classes_seen_so_far}")
            print(f"Training on train {len(train_experience.dataset)} examples")
            print(f"Experience number validation {val_experience.current_experience}")
            print(f"Classes in this experience val {val_experience.classes_in_this_experience}")
            print(f"Classes seen so far validation {val_experience.classes_seen_so_far}")
            print(f"Validating on validation {len(val_experience.dataset)} examples")
            # print(f"Experience number Acc validation {accumulated_val_dataset.current_experience}")
            # print(f"Classes in this experience Acc val {accumulated_val_dataset.classes_in_this_experience}")
            # print(f"Classes seen so far Acc  validation {accumulated_val_dataset.classes_seen_so_far}")
            # print(f"Validating on Acc validation {len(accumulated_val_dataset.dataset)} examples")

            cl_strategy.train(train_experience, eval_streams=[val_experience])

        cl_strategy.eval(test_stream)


        loggers = []
        loggers.append(WandBLogger(project_name=project_name, run_name="CWRStar", params={"reinit": True, "group": group_name}))
        loggers.append(InteractiveLogger())
        loggers.append(TextLogger(open(f'CWRStar_{i}.txt', 'a')))

        eval_plugin = EvaluationPlugin(
        accuracy_metrics(epoch=True, stream=True, trained_experience=True),
        class_accuracy_metrics(epoch=True, stream=True),
        cpu_usage_metrics(epoch=True),
        gpu_usage_metrics(gpu_id=0, epoch=True),
        ram_usage_metrics(epoch=True),
        disk_usage_metrics(epoch=True),
        # forward_transfer_metrics(stream=True),
        forgetting_metrics(stream=True),
        loggers=loggers,
        strict_checks=False)
        
        cl_strategy = CWRStar(
            model, optimizer, criterion, cwr_layer_name=None, device=device,
            train_mb_size=batchsize_train, train_epochs=epochs, eval_mb_size=batchsize_eval, evaluator=eval_plugin, \
            plugins=[EarlyStoppingPlugin(patience, "valid", "Top1_Acc_Stream")], eval_every=eval_every)

        print(f"Current training strategy: {cl_strategy}")
        for train_experience, val_experience in zip(train_stream, val_stream):
            print(f"Experience number train {train_experience.current_experience}")
            print(f"classes in this experience train {train_experience.classes_in_this_experience}")
            print(f"Classes seen so far train {train_experience.classes_seen_so_far}")
            print(f"Training on train {len(train_experience.dataset)} examples")
            print(f"Experience number validation {val_experience.current_experience}")
            print(f"Classes in this experience val {val_experience.classes_in_this_experience}")
            print(f"Classes seen so far validation {val_experience.classes_seen_so_far}")
            print(f"Validating on validation {len(val_experience.dataset)} examples") 

            cl_strategy.train(train_experience, eval_streams=[val_experience])

        cl_strategy.eval(test_stream)


        loggers = []
        loggers.append(WandBLogger(project_name=project_name, run_name="GEM", params={"reinit": True, "group": group_name}))
        loggers.append(InteractiveLogger())
        loggers.append(TextLogger(open(f'GEM_{i}.txt', 'a')))

        eval_plugin = EvaluationPlugin(
        accuracy_metrics(epoch=True, stream=True, trained_experience=True),
        class_accuracy_metrics(epoch=True, stream=True),
        cpu_usage_metrics(epoch=True),
        gpu_usage_metrics(gpu_id=0, epoch=True),
        ram_usage_metrics(epoch=True),
        disk_usage_metrics(epoch=True),
        # forward_transfer_metrics(stream=True),
        forgetting_metrics(stream=True),
        loggers=loggers,
        strict_checks=False)

        cl_strategy = GEM(
            model, optimizer, criterion, device=device, patterns_per_exp=256, memory_strength=0.5,
            train_mb_size=batchsize_train, train_epochs=epochs, eval_mb_size=batchsize_eval, evaluator=eval_plugin,
            plugins=[EarlyStoppingPlugin(patience, "valid", "Top1_Acc_Stream")], eval_every=eval_every)

        print(f"Current training strategy: {cl_strategy}")
        for train_experience, val_experience in zip(train_stream, val_stream):
            print(f"Experience number train {train_experience.current_experience}")
            print(f"classes in this experience train {train_experience.classes_in_this_experience}")
            print(f"Classes seen so far train {train_experience.classes_seen_so_far}")
            print(f"Training on train {len(train_experience.dataset)} examples")
            print(f"Experience number validation {val_experience.current_experience}")
            print(f"Classes in this experience val {val_experience.classes_in_this_experience}")
            print(f"Classes seen so far validation {val_experience.classes_seen_so_far}")
            print(f"Validating on validation {len(val_experience.dataset)} examples") 
        
            cl_strategy.train(train_experience, eval_streams=[val_experience])

        cl_strategy.eval(test_stream)
        
        loggers = []
        loggers.append(WandBLogger(project_name=project_name, run_name="EWC", params={"reinit": True, "group": group_name}))
        loggers.append(InteractiveLogger())
        loggers.append(TextLogger(open(f'EWC_{i}.txt', 'a')))

        eval_plugin = EvaluationPlugin(
        accuracy_metrics(epoch=True, stream=True, trained_experience=True),
        class_accuracy_metrics(epoch=True, stream=True),
        cpu_usage_metrics(epoch=True),
        gpu_usage_metrics(gpu_id=0, epoch=True),
        ram_usage_metrics(epoch=True),
        disk_usage_metrics(epoch=True),
        # forward_transfer_metrics(stream=True),
        forgetting_metrics(stream=True),
        loggers=loggers,
        strict_checks=False)

        cl_strategy = EWC(
            model, optimizer, criterion, ewc_lambda=1.0, device=device,
            train_mb_size=batchsize_train, train_epochs=epochs, eval_mb_size=batchsize_eval, evaluator=eval_plugin, \
            plugins=[EarlyStoppingPlugin(patience, "valid", "Top1_Acc_Stream")], eval_every=eval_every)

        print(f"Current training strategy: {cl_strategy}")
        for train_experience, val_experience in zip(train_stream, val_stream):
            print(f"Experience number train {train_experience.current_experience}")
            print(f"classes in this experience train {train_experience.classes_in_this_experience}")
            print(f"Classes seen so far train {train_experience.classes_seen_so_far}")
            print(f"Training on train {len(train_experience.dataset)} examples")
            print(f"Experience number validation {val_experience.current_experience}")
            print(f"Classes in this experience val {val_experience.classes_in_this_experience}")
            print(f"Classes seen so far validation {val_experience.classes_seen_so_far}")
            print(f"Validating on validation {len(val_experience.dataset)} examples")  

            cl_strategy.train(train_experience, eval_streams=[val_experience])

        cl_strategy.eval(test_stream)

        loggers = []
        loggers.append(WandBLogger(project_name=project_name, run_name="Cumulative", params={"reinit": True, "group": group_name}))
        loggers.append(InteractiveLogger())
        loggers.append(TextLogger(open(f'Cumulative_{i}.txt', 'a')))

        eval_plugin = EvaluationPlugin(
        accuracy_metrics(epoch=True, stream=True, trained_experience=True),
        class_accuracy_metrics(epoch=True, stream=True),
        cpu_usage_metrics(epoch=True),
        gpu_usage_metrics(gpu_id=0, epoch=True),
        ram_usage_metrics(epoch=True),
        disk_usage_metrics(epoch=True),
        # forward_transfer_metrics(stream=True),
        forgetting_metrics(stream=True),
        loggers=loggers,
        strict_checks=False)


        cl_strategy = Cumulative(
            model, optimizer, criterion, device=device,
            train_mb_size=batchsize_train, train_epochs=epochs, eval_mb_size=batchsize_eval, evaluator=eval_plugin, \
            plugins=[EarlyStoppingPlugin(patience, "valid", "Top1_Acc_Stream")], eval_every=eval_every)

        print(f"Current training strategy: {cl_strategy}")
        for train_experience, val_experience in zip(train_stream, val_stream):
            print(f"Experience number train {train_experience.current_experience}")
            print(f"classes in this experience train {train_experience.classes_in_this_experience}")
            print(f"Classes seen so far train {train_experience.classes_seen_so_far}")
            print(f"Training on train {len(train_experience.dataset)} examples")
            print(f"Experience number validation {val_experience.current_experience}")
            print(f"Classes in this experience val {val_experience.classes_in_this_experience}")
            print(f"Classes seen so far validation {val_experience.classes_seen_so_far}")
            print(f"Validating on validation {len(val_experience.dataset)} examples")  

            cl_strategy.train(train_experience, eval_streams=[val_experience])

        cl_strategy.eval(test_stream)

    # Finish the current WandB run
    wandb.finish()


def train_iteratively_without_ES(project_name, num_runs=5):

    # Load the CORe50 dataset
    core50 = CORe50(scenario="nc", mini=True, object_lvl=True)
    core50 = benchmark_with_validation_stream(core50, 0.2)
    train_stream = core50.train_stream
    val_stream = core50.valid_stream
    test_stream = core50.test_stream

    if torch.cuda.is_available():
        device = 'cuda:0'
        torch.backends.cudnn.benchmark = True
        device_count = torch.cuda.device_count()
        print(f"Found {device_count} CUDA GPU devices.")
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    print(f'Using {device} device')

    # Parameters
    epochs = 500
    batchsize_train = 100
    batchsize_eval = 100
    patience = 5
    eval_every = 1

    # Define list of strategies
    strategies = ['Naive', 'CWRStar', 'GEM', 'EWC', 'Cumulative']  # Add the names of other strategies here.

    # Iterate through different strategies
    for run_idx in range(num_runs):
        for strategy_name in strategies:
            # Reset the model and optimizer for each strategy
            model = SimpleCNN(num_classes=50).to(device)
            optimizer = Adam(model.parameters(), lr=0.001)
            criterion = CrossEntropyLoss()

            # Configure WandB logger
            wandb.init(project=project_name, name=strategy_name)

            loggers = [WandBLogger(), InteractiveLogger(), TensorboardLogger(), TextLogger(open(f'log_{strategy_name}.txt', 'a'))]

            # Configure evaluation plugin
            eval_plugin = EvaluationPlugin(
                accuracy_metrics(epoch=True, stream=True),
                class_accuracy_metrics(epoch=True, stream=True),
                cpu_usage_metrics(epoch=True),
                gpu_usage_metrics(gpu_id=0, epoch=True),
                ram_usage_metrics(epoch=True),
                disk_usage_metrics(epoch=True),
                forgetting_metrics(stream=True),
                loggers=loggers,
                strict_checks=False)
            
            # Select strategy
            if strategy_name == 'Naive':
                cl_strategy = Naive(
                    model, optimizer, criterion, device=device,
                    train_mb_size=batchsize_train, train_epochs=epochs, eval_mb_size=batchsize_eval,
                    evaluator=eval_plugin, eval_every=eval_every)
            elif strategy_name == 'CWRStar':
                cl_strategy = CWRStar(
                    model, optimizer, criterion, cwr_layer_name=None, device=device,
                    train_mb_size=batchsize_train, train_epochs=epochs, eval_mb_size=batchsize_eval,
                    evaluator=eval_plugin, eval_every=eval_every)
            elif strategy_name == 'GEM':
                cl_strategy = GEM(
                    model, optimizer, criterion, device=device, patterns_per_exp=256, memory_strength=0.5,
                    train_mb_size=batchsize_train, train_epochs=epochs, eval_mb_size=batchsize_eval,
                    evaluator=eval_plugin, eval_every=eval_every) 
            elif strategy_name == 'EWC':
                cl_strategy = EWC(
                    model, optimizer, criterion, ewc_lambda=1.0, device=device,
                    train_mb_size=batchsize_train, train_epochs=epochs, eval_mb_size=batchsize_eval,
                    evaluator=eval_plugin, eval_every=eval_every) 
            elif strategy_name == 'Cumulative':
                cl_strategy = Cumulative(
                    model, optimizer, criterion, device=device,
                    train_mb_size=batchsize_train, train_epochs=epochs, eval_mb_size=batchsize_eval,
                    evaluator=eval_plugin, eval_every=eval_every) 
                        
            
            print(f"Current training strategy: {cl_strategy}")

            # Train and evaluate
            for train_experience, val_experience in zip(train_stream, val_stream):
                print(f"Experience number train {train_experience.current_experience}")
                print(f"Classes seen so far train {train_experience.classes_seen_so_far}")
                print(f"Training on train {len(train_experience.dataset)} examples")
                print(f"Experience number validation {val_experience.current_experience}")
                print(f"Classes seen so far validation {val_experience.classes_seen_so_far}")
                print(f"Validating on validation {len(val_experience.dataset)} examples")    

                cl_strategy.train(train_experience, eval_streams=[val_experience])

            # cl_strategy.eval(test_stream)
            
            # Finish the current WandB run
            wandb.finish()

def train_iteratively_ES(project_name, num_runs):

    # Load the CORe50 dataset
    core50 = CORe50(scenario="nc", mini=True, object_lvl=True)
    core50 = benchmark_with_validation_stream(core50, 0.2)
    train_stream = core50.train_stream
    val_stream = core50.valid_stream

    if torch.cuda.is_available():
        device = 'cuda:0'
        torch.backends.cudnn.benchmark = True
        device_count = torch.cuda.device_count()
        print(f"Found {device_count} CUDA GPU devices.")
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    print(f'Using {device} device')

    # Parameters
    epochs = 500
    batchsize_train = 100
    batchsize_eval = 100
    patience = PATIENCE
    eval_every = 1

    # Define list of strategies
    strategies = ['Naive', 'CWRStar', 'GEM', 'EWC', 'Cumulative']  # Add the names of other strategies here.

    # Iterate through different strategies
    for run_idx in range(num_runs):
        for strategy_name in strategies:
            # Reset the model and optimizer for each strategy
            model = SimpleCNN(num_classes=50).to(device)
            optimizer = Adam(model.parameters(), lr=0.001)
            criterion = CrossEntropyLoss()

            # Configure WandB logger
            wandb.init(project=project_name, name=strategy_name)

            loggers = [WandBLogger(), InteractiveLogger(), TensorboardLogger(), TextLogger(open(f'log_{strategy_name}.txt', 'a'))]

            # Configure evaluation plugin
            eval_plugin = EvaluationPlugin(
                accuracy_metrics(epoch=True, stream=True),
                class_accuracy_metrics(epoch=True, stream=True),
                cpu_usage_metrics(epoch=True),
                gpu_usage_metrics(gpu_id=0, epoch=True),
                ram_usage_metrics(epoch=True),
                disk_usage_metrics(epoch=True),
                forgetting_metrics(stream=True),
                loggers=loggers,
                strict_checks=False)
            
            # Select strategy
            if strategy_name == 'Naive':
                cl_strategy = Naive(
                    model, optimizer, criterion, device=device,
                    train_mb_size=batchsize_train, train_epochs=epochs, eval_mb_size=batchsize_eval,
                    evaluator=eval_plugin, eval_every=eval_every,
                    plugins=[EarlyStoppingPlugin(patience, "valid_stream")])
            elif strategy_name == 'CWRStar':
                cl_strategy = CWRStar(
                    model, optimizer, criterion, cwr_layer_name=None, device=device,
                    train_mb_size=batchsize_train, train_epochs=epochs, eval_mb_size=batchsize_eval,
                    evaluator=eval_plugin, eval_every=eval_every,
                    plugins=[EarlyStoppingPlugin(patience, "valid_stream")])
            elif strategy_name == 'GEM':
                cl_strategy = GEM(
                    model, optimizer, criterion, device=device, patterns_per_exp=256, memory_strength=0.5,
                    train_mb_size=batchsize_train, train_epochs=epochs, eval_mb_size=batchsize_eval,
                    evaluator=eval_plugin, eval_every=eval_every,
                    plugins=[EarlyStoppingPlugin(patience, "valid_stream")]) 
            elif strategy_name == 'EWC':
                cl_strategy = EWC(
                    model, optimizer, criterion, ewc_lambda=1.0, device=device,
                    train_mb_size=batchsize_train, train_epochs=epochs, eval_mb_size=batchsize_eval,
                    evaluator=eval_plugin, eval_every=eval_every,
                    plugins=[EarlyStoppingPlugin(patience, "valid_stream")]) 
            elif strategy_name == 'Cumulative':
                cl_strategy = Cumulative(
                    model, optimizer, criterion, device=device,
                    train_mb_size=batchsize_train, train_epochs=epochs, eval_mb_size=batchsize_eval,
                    evaluator=eval_plugin, eval_every=eval_every,
                    plugins=[EarlyStoppingPlugin(patience, "valid_stream")]) 
                        
            
            print(f"Current training strategy: {cl_strategy}")

            # Train and evaluate
            for train_experience, val_experience in zip(train_stream, val_stream):
                print(f"Experience number train {train_experience.current_experience}")
                print(f"Classes seen so far train {train_experience.classes_seen_so_far}")
                print(f"Training on train {len(train_experience.dataset)} examples")
                print(f"Experience number validation {val_experience.current_experience}")
                print(f"Classes seen so far validation {val_experience.classes_seen_so_far}")
                print(f"Validating on validation {len(val_experience.dataset)} examples")    

                cl_strategy.train(train_experience, eval_streams=[val_experience])

            # Finish the current WandB run
            wandb.finish()


def extract_metrics(project_name):
    # Login to the wandb
    wandb.login()
    # Extract the metrics from WandB after all the runs
    api = wandb.Api()
    runs = api.runs(project_name)

    # Initialize empty DataFrame
    data = pd.DataFrame()

    # Fetch the logged metrics for each run
    for run in runs:
        history = run.history(samples=10000)
        history['run_id'] = run.id
        history['strategy_name'] = run.name
        data = pd.concat([data, history], ignore_index=True)

    data.to_excel("output1.xlsx", index=False)

    # Add a helper column for the order of appearance within each strategy group
    data['order'] = data.groupby(['strategy_name']).cumcount() // 5

    # Group by 'strategy' and 'order', then calculate mean accuracy and variance
    grouped = data.groupby(['strategy_name', 'order'])['Top1_Acc_Epoch/train_phase/train_stream/Task000']
    mean_accuracies = grouped.mean().reset_index()
    variances = grouped.var().reset_index()

    # Pivot the table to have strategies as columns
    result_means = mean_accuracies.pivot(index='order', columns='strategy_name', values='Top1_Acc_Epoch/train_phase/train_stream/Task000')
    result_variances = variances.pivot(index='order', columns='strategy_name', values='Top1_Acc_Epoch/train_phase/train_stream/Task000')

    # Convert the pivoted DataFrame to a dictionary with strategies as keys and lists of means as values
    result_means_dict = result_means.to_dict(orient='list')
    result_variances_dict = result_variances.to_dict(orient='list')

    # Plotting
    plt.figure(figsize=(12, 6))

    # Define colors and line styles for each strategy
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    linestyles = ['-', '--', '-.', ':']

    # Loop through each strategy
    for i, strategy in enumerate(result_means_dict.keys()):
        means = result_means_dict[strategy]
        std_dev = np.sqrt(result_variances_dict[strategy])
        x = range(1, len(means) + 1)
        
        # Plot means for this strategy with customizations
        plt.plot(x, means,
                 label=strategy,
                 linewidth=2,                 # Increase line width
                 color=colors[i % len(colors)], # Set different colors
                 linestyle=linestyles[i % len(linestyles)], # Set different line styles
                 marker='o')                  # Use markers for each point
        
        # Add variance as shadowed region
        plt.fill_between(x, np.subtract(means, std_dev), np.add(means, std_dev),
                         color=colors[i % len(colors)], alpha=0.2)

    # Adding labels and title
    plt.xlabel('Epochs')
    plt.ylabel('Mean Accuracy')
    plt.title('Mean Accuracies for Different Strategies with Variance')
    plt.legend()

    # Show plot
    plt.show()

def extract_accuracy_valid(project_name):
    # Login to the wandb
    wandb.login()
    # Extract the metrics from WandB after all the runs
    api = wandb.Api()
    runs = api.runs(project_name)

    # Initialize empty DataFrame
    data = pd.DataFrame()

    # Fetch the logged metrics for each run
    for run in runs:
        history = run.history(samples=10000)
        history['run_id'] = run.id
        history['strategy_name'] = run.name
        data = pd.concat([data, history], ignore_index=True)

    data.to_excel("output_acc_valid.xlsx", index=False)

    data = data["run_id"].unique()

    data.to_excel("output_acc_valid2.xlsx", index=False)



def extract_convergence(project_name, num_runs):
    # Login to the wandb
    wandb.login()
    # Extract the metrics from WandB after all the runs
    api = wandb.Api()
    runs = api.runs(project_name)

    # Initialize empty DataFrame
    df = pd.DataFrame()

    counts = []
    counter = 0
    strategy_names = []
    # Fetch the logged metrics for each run
    for run in runs:
        history = run.history()
        history = pd.DataFrame(history)
        counts = [] 
        strategy_names.append(run.name)
        for index, row in history.iterrows():
            if str(row['TrainingExperience']).split('.')[0].isdigit():
                counter += 1
                # Append the count to the list and reset the counter
                counts.append(counter)
                counter = 0
            else:
                counter += 1
        # print(len(counts))
        df[run.name + str(strategy_names.count(run.name))] = counts

    print("COUNTS", df.head(10))

    history.to_excel("convergence_output.xlsx")

    # Strategies names without version numbers
    strategies = ['Cumulative', 'EWC', 'GEM', 'CWRStar', 'Naive']

    # Calculate the mean and standard deviation for each strategy
    summary = {}
    fig, ax = plt.subplots(figsize=(12, 6))
    for strategy in strategies:
        # For mean and standard deviation of sum of epochs
        columns = [col for col in df.columns if col.startswith(strategy)]
        strategy_data = df[columns]
        values = strategy_data.values - PATIENCE
        mean = values.mean()
        std = values.std()
        summary[strategy] = {'mean': mean, 'std': std}


        # For number of epochs
        strategy_data = []
        for i in range(num_runs):
            column_name = f'{strategy}{i}'
            if column_name in df:
                strategy_data.extend(df[column_name] - PATIENCE)
        plt.plot(range(len(strategy_data)), strategy_data, marker='o', label=strategy)

    # For number of epochs
    # plt.axhline(y=PATIENCE, color='red', linewidth=2, linestyle='--', label='Patience')
    plt.xlabel('Experience')
    plt.ylabel('Number of epochs')
    plt.title('Number of epochs in each experience for class incremental scenario')
    # Set custom tick labels
    total_iterations = num_runs * 9  # 10 runs, 9 counts for each strategy
    tick_positions = list(range(total_iterations))
    tick_labels = [str(i % 9) for i in range(total_iterations)]
    plt.xticks(tick_positions, tick_labels)
    plt.xticks(range(len(strategy_data)))
    plt.legend()

    # Show plot
    plt.tight_layout()
    plt.grid(True)
    plt.savefig("epochs.png")


    # For mean and standard deviation of sum of epochs
    # Convert the summary to a DataFrame for better visualization
    summary_df = pd.DataFrame(summary).T

    # Plot
    plt.figure(figsize=(10, 5))
    bars = plt.bar(summary_df.index, summary_df['mean'], yerr=summary_df['std'], capsize=7, color='skyblue', edgecolor='black')

    # Adding titles and labels
    plt.title('Mean and Standard Deviation of Different Strategies for Class Incremental Scenario')
    plt.xlabel('Strategy')
    plt.ylabel('Value')

    # Adding numerical values at the top of each bar
    for bar, value in zip(bars, summary_df['mean']):
        plt.annotate(f'{value:.2f}', xy=(bar.get_x() + bar.get_width() / 3, value),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

    # Show plot
    plt.savefig("strategies.png")

    # Print the summary DataFrame
    print(summary_df)



    print("finished extracting convergence")


def train_without_ES(group_name):

    # Load the CORe50 dataset
    core50 = CORe50(scenario="nc", mini=True, object_lvl=True)

    # the task label of each train_experience.
    print('--- Task labels:')
    print(core50.task_labels)

    # Instatiate train and test streams
    train_stream = core50.train_stream
    test_stream = core50.test_stream


    if torch.cuda.is_available():
        device = 'cuda:0'
        torch.backends.cudnn.benchmark = True
        device_count = torch.cuda.device_count()
        print(f"Found {device_count} CUDA GPU devices.")
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    print(f'Using {device} device')

    model = SimpleCNN(num_classes=50).to(device)
    # model = SimpleMLP(num_classes=50, input_size=32*32*3, hidden_size=32*32*3, hidden_layers=4, drop_rate=0.5).to(device)
    # pnn_model = PNN(num_layers=4, in_features=32*32*3, hidden_features_per_column=32*32*3).to(device)
    # gr_model = MlpVAE((3, 32, 32), nhid=2, n_classes=50, device=device)
    print(f"Main model {model}")
    # print(f"PNN model {pnn_model}")
    # print(f"GR model {gr_model}")
    # model = MlpVAE((3, 32, 32), nhid=2, device=device)

    # model = MTSimpleCNN().to(device)

    # model = SimpleMLP(num_classes=50, input_size=(3,32,32), hidden_size=32, hidden_layers = 2, drop_rate=0.5)

    # model = SlimResNet18(nclasses=50)

    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = CrossEntropyLoss()
    epochs = 80
    batchsize_train = 100
    batchsize_eval = 100

    loggers = []
    loggers.append(WandBLogger(project_name="avalanche", run_name="Naive", params={"reinit": True, "group": group_name}))
    loggers.append(InteractiveLogger())
    loggers.append(TextLogger(open('log.txt', 'a')))

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(epoch=True, stream=True),
        class_accuracy_metrics(epoch=True, stream=True),
        cpu_usage_metrics(epoch=True),
        gpu_usage_metrics(gpu_id=0, epoch=True),
        ram_usage_metrics(epoch=True),
        disk_usage_metrics(epoch=True),
        # forward_transfer_metrics(stream=True),
        forgetting_metrics(stream=True),
        loggers=loggers,
        strict_checks=False)
    
    cl_strategy = Naive(
        model, optimizer, criterion, device=device,
        train_mb_size=batchsize_train, train_epochs=epochs, eval_mb_size=batchsize_eval, evaluator=eval_plugin)
       
    print(f"Current training strategy: {cl_strategy}")
    for train_experience in train_stream:
        print(f"Experience number {train_experience.current_experience}")
        print(f"Classes seen so far {train_experience.classes_seen_so_far}")
        print(f"Training on {len(train_experience.dataset)} examples")

        cl_strategy.train(train_experience)
        cl_strategy.eval(test_stream)
 

    loggers = []
    loggers.append(WandBLogger(project_name="avalanche", run_name="CWRStar", params={"reinit": True, "group": group_name}))
    loggers.append(InteractiveLogger())
    loggers.append(TextLogger(open('log.txt', 'a')))

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(epoch=True, stream=True),
        class_accuracy_metrics(epoch=True, stream=True),
        cpu_usage_metrics(epoch=True),
        gpu_usage_metrics(gpu_id=0, epoch=True),
        ram_usage_metrics(epoch=True),
        disk_usage_metrics(epoch=True),
        # forward_transfer_metrics(stream=True),
        forgetting_metrics(stream=True),
        loggers=loggers,
        strict_checks=False
    )
    
    cl_strategy = CWRStar(
        model, optimizer, criterion, cwr_layer_name=None, device=device,
        train_mb_size=batchsize_train, train_epochs=epochs, eval_mb_size=batchsize_eval, evaluator=eval_plugin)

    print(f"Current training strategy: {cl_strategy}")
    for train_experience in train_stream:
        print(f"Experience number {train_experience.current_experience}")
        print(f"Classes seen so far {train_experience.classes_seen_so_far}")
        print(f"Training on {len(train_experience.dataset)} examples")

        cl_strategy.train(train_experience)


    cl_strategy.eval(test_stream)

    loggers = []
    loggers.append(WandBLogger(project_name="avalanche", run_name="GEM", params={"reinit": True, "group": group_name}))
    loggers.append(InteractiveLogger())
    loggers.append(TextLogger(open('log.txt', 'a')))

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(epoch=True, stream=True),
        class_accuracy_metrics(epoch=True, stream=True),
        cpu_usage_metrics(epoch=True),
        gpu_usage_metrics(gpu_id=0, epoch=True),
        ram_usage_metrics(epoch=True),
        disk_usage_metrics(epoch=True),
        # forward_transfer_metrics(stream=True),
        forgetting_metrics(stream=True),
        loggers=loggers,
        strict_checks=False
    )
    
    cl_strategy = GEM(
        model, optimizer, criterion, device=device, patterns_per_exp=256, memory_strength=0.5,
        train_mb_size=batchsize_train, train_epochs=epochs, eval_mb_size=batchsize_eval, evaluator=eval_plugin)

    print(f"Current training strategy: {cl_strategy}")
    for train_experience in train_stream:
        print(f"Experience number {train_experience.current_experience}")
        print(f"Classes seen so far {train_experience.classes_seen_so_far}")
        print(f"Training on {len(train_experience.dataset)} examples")

        cl_strategy.train(train_experience)

    cl_strategy.eval(test_stream)
    
    loggers = []
    loggers.append(WandBLogger(project_name="avalanche", run_name="EWC", params={"reinit": True, "group": group_name}))
    loggers.append(InteractiveLogger())
    loggers.append(TextLogger(open('log.txt', 'a')))

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(epoch=True, stream=True),
        class_accuracy_metrics(epoch=True, stream=True),
        cpu_usage_metrics(epoch=True),
        gpu_usage_metrics(gpu_id=0, epoch=True),
        ram_usage_metrics(epoch=True),
        disk_usage_metrics(epoch=True),
        # forward_transfer_metrics(stream=True),
        forgetting_metrics(stream=True),
        loggers=loggers,
        strict_checks=False
    )

    cl_strategy = EWC(
        model, optimizer, criterion, ewc_lambda=1.0, device=device,
        train_mb_size=batchsize_train, train_epochs=epochs, eval_mb_size=batchsize_eval, evaluator=eval_plugin)

    print(f"Current training strategy: {cl_strategy}")
    for train_experience in train_stream:
        print(f"Experience number {train_experience.current_experience}")
        print(f"Classes seen so far {train_experience.classes_seen_so_far}")
        print(f"Training on {len(train_experience.dataset)} examples")

        cl_strategy.train(train_experience)

    cl_strategy.eval(test_stream)

    loggers = []
    loggers.append(WandBLogger(project_name="avalanche", run_name="Cumulative", params={"reinit": True, "group": group_name}))
    loggers.append(InteractiveLogger())
    loggers.append(TextLogger(open('log.txt', 'a')))

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(epoch=True, stream=True),
        class_accuracy_metrics(epoch=True, stream=True),
        cpu_usage_metrics(epoch=True),
        gpu_usage_metrics(gpu_id=0, epoch=True),
        ram_usage_metrics(epoch=True),
        disk_usage_metrics(epoch=True),
        # forward_transfer_metrics(stream=True),
        forgetting_metrics(stream=True),
        loggers=loggers,
    )

    cl_strategy = Cumulative(
        model, optimizer, criterion, device=device,
        train_mb_size=batchsize_train, train_epochs=epochs, eval_mb_size=batchsize_eval, evaluator=eval_plugin)

    print(f"Current training strategy: {cl_strategy}")
    for train_experience in train_stream:
        print(f"Experience number {train_experience.current_experience}")
        print(f"Classes seen so far {train_experience.classes_seen_so_far}")
        print(f"Training on {len(train_experience.dataset)} examples")

        cl_strategy.train(train_experience)



    cl_strategy.eval(test_stream)

def main3(group_name):

    # Load the CORe50 dataset
    core50 = CORe50(scenario="nc", mini=True, object_lvl=True)

    # the task label of each train_experience.
    print('--- Task labels:')
    print(core50.task_labels)

    # Instatiate train and test streams
    train_stream = core50.train_stream
    test_stream = core50.test_stream


    if torch.cuda.is_available():
        device = 'cuda:0'
        torch.backends.cudnn.benchmark = True
        device_count = torch.cuda.device_count()
        print(f"Found {device_count} CUDA GPU devices.")
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    print(f'Using {device} device')

    model = SimpleCNN(num_classes=50).to(device)
    # model = SimpleMLP(num_classes=50, input_size=32*32*3, hidden_size=32*32*3, hidden_layers=4, drop_rate=0.5).to(device)
    # pnn_model = PNN(num_layers=4, in_features=32*32*3, hidden_features_per_column=32*32*3).to(device)
    # gr_model = MlpVAE((3, 32, 32), nhid=2, n_classes=50, device=device)
    print(f"Main model {model}")
    # print(f"PNN model {pnn_model}")
    # print(f"GR model {gr_model}")
    # model = MlpVAE((3, 32, 32), nhid=2, device=device)

    # model = MTSimpleCNN().to(device)

    # model = SimpleMLP(num_classes=50, input_size=(3,32,32), hidden_size=32, hidden_layers = 2, drop_rate=0.5)

    # model = SlimResNet18(nclasses=50)

    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = CrossEntropyLoss()
    epochs = 20
    batchsize_train = 100
    batchsize_eval = 100

    loggers = []
    loggers.append(WandBLogger(project_name="avalanche", run_name="Naive", params={"reinit": True, "group": group_name}))
    loggers.append(InteractiveLogger())
    loggers.append(TextLogger(open('log.txt', 'a')))

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(epoch=True, stream=True),
        class_accuracy_metrics(epoch=True, stream=True),
        cpu_usage_metrics(epoch=True),
        gpu_usage_metrics(gpu_id=0, epoch=True),
        ram_usage_metrics(epoch=True),
        disk_usage_metrics(epoch=True),
        forward_transfer_metrics(stream=True),
        forgetting_metrics(stream=True),
        loggers=loggers,
        strict_checks=False)
    
    # Strategies
    ewc = EWCPlugin(ewc_lambda=0.001)


    strategy = SupervisedTemplate(
        model, optimizer, criterion,
        plugins=[eval_plugin, ewc])



def gem(it=5, start_ppe=4):

    # Load the CORe50 dataset
    core50 = CORe50(scenario="nc", mini=True, object_lvl=True)

    # the task label of each train_experience.
    print('--- Task labels:')
    print(core50.task_labels)

    # Instatiate train and test streams
    train_stream = core50.train_stream
    test_stream = core50.test_stream


    if torch.cuda.is_available():
        device = 'cuda:0'
        torch.backends.cudnn.benchmark = True
        device_count = torch.cuda.device_count()
        print(f"Found {device_count} CUDA GPU devices.")
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    print(f'Using {device} device')

    model = SimpleMLP(num_classes=50, input_size=32*32*3, hidden_size=32*32*3, hidden_layers=4, drop_rate=0.5).to(device)
    # pnn_model = PNN(num_layers=4, in_features=32*32*3, hidden_features_per_column=32*32*3).to(device)
    # gr_model = MlpVAE((3, 32, 32), nhid=2, n_classes=50, device=device)
    print(f"Main model {model}")
    # print(f"PNN model {pnn_model}")
    # print(f"GR model {gr_model}")
    # model = MlpVAE((3, 32, 32), nhid=2, device=device)

    # model = MTSimpleCNN().to(device)

    # model = SimpleMLP(num_classes=50, input_size=(3,32,32), hidden_size=32, hidden_layers = 2, drop_rate=0.5)

    # model = SlimResNet18(nclasses=50)

    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = CrossEntropyLoss()
    epochs = 1
    batchsize_train = 100
    batchsize_eval = 100

    for i in range(2, 2*it, 2):
        patterns_per_experience = i*start_ppe
        
        run_name = f"GEM_patterns_per_experience_{patterns_per_experience}"
        logger = WandBLogger(project_name="avalanche", run_name=run_name, params={"reinit": True, "group": "GEM"})

        eval_plugin = EvaluationPlugin(
            accuracy_metrics(epoch=True),
            cpu_usage_metrics(epoch=True),
            gpu_usage_metrics(gpu_id=0, epoch=True),
            ram_usage_metrics(epoch=True),
            disk_usage_metrics(epoch=True),
            loggers=logger,
            strict_checks=False
        )

        print(f"Patterns per train_experience: {patterns_per_experience}")
        cl_strategy = GEM(
            model, optimizer, criterion, device=device,
            train_mb_size=batchsize_train, train_epochs=epochs, eval_mb_size=batchsize_eval, evaluator=eval_plugin,
            patterns_per_exp=patterns_per_experience
        )

        print(f"Current training strategy: {cl_strategy}")
        results = []
        for train_experience in train_stream:
            print(f"Experience number {train_experience.current_experience}")
            print(f"Classes seen so far {train_experience.classes_seen_so_far}")
            print(f"Training on {len(train_experience.dataset)} examples")

            cl_strategy.train(train_experience)

        # Evaluate on test set
        print(f"Testing on {len(test_stream[0].dataset)} examples")

        print(f"Evaluation on test stream {cl_strategy.eval(test_stream)}")



def ewc(it=5, ewc_lambda=0.25):

    # Load the CORe50 dataset
    core50 = CORe50(scenario="nc", mini=True, object_lvl=True)

    # the task label of each train_experience.
    print('--- Task labels:')
    print(core50.task_labels)

    # Instatiate train and test streams
    train_stream = core50.train_stream
    test_stream = core50.test_stream


    if torch.cuda.is_available():
        device = 'cuda:0'
        torch.backends.cudnn.benchmark = True
        device_count = torch.cuda.device_count()
        print(f"Found {device_count} CUDA GPU devices.")
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    print(f'Using {device} device')

    model = SimpleMLP(num_classes=50, input_size=32*32*3, hidden_size=32*32*3, hidden_layers=4, drop_rate=0.5).to(device)
    # pnn_model = PNN(num_layers=4, in_features=32*32*3, hidden_features_per_column=32*32*3).to(device)
    # gr_model = MlpVAE((3, 32, 32), nhid=2, n_classes=50, device=device)
    print(f"Main model {model}")
    # print(f"PNN model {pnn_model}")
    # print(f"GR model {gr_model}")
    # model = MlpVAE((3, 32, 32), nhid=2, device=device)

    # model = MTSimpleCNN().to(device)

    # model = SimpleMLP(num_classes=50, input_size=(3,32,32), hidden_size=32, hidden_layers = 2, drop_rate=0.5)

    # model = SlimResNet18(nclasses=50)

    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = CrossEntropyLoss()
    epochs = 1
    batchsize_train = 100
    batchsize_eval = 100

    for i in range(2, 2*it, 2):
        EWC_lambda = i*ewc_lambda
        
        run_name = f"EWC_lambda{EWC_lambda}"
        logger = WandBLogger(project_name="avalanche", run_name=run_name, params={"reinit": True, "group": "EWC"})

        eval_plugin = EvaluationPlugin(
            accuracy_metrics(epoch=True),
            cpu_usage_metrics(epoch=True),
            gpu_usage_metrics(gpu_id=0, epoch=True),
            ram_usage_metrics(epoch=True),
            disk_usage_metrics(epoch=True),
            loggers=logger,
            strict_checks=False
        )

        print(f"EWC lambda: {EWC_lambda}")
        cl_strategy = GEM(
            model, optimizer, criterion, device=device,
            train_mb_size=batchsize_train, train_epochs=epochs, eval_mb_size=batchsize_eval, evaluator=eval_plugin,
            ewc_lambda = EWC_lambda
        )

        print(f"Current training strategy: {cl_strategy}")
        results = []
        for train_experience in train_stream:
            print(f"Experience number {train_experience.current_experience}")
            print(f"Classes seen so far {train_experience.classes_seen_so_far}")
            print(f"Training on {len(train_experience.dataset)} examples")

            cl_strategy.train(train_experience)

        # Evaluate on test set
        print(f"Testing on {len(test_stream[0].dataset)} examples")

        print(f"Evaluation on test stream {cl_strategy.eval(test_stream)}")




def GR_GR():
    from avalanche.benchmarks import SplitMNIST
    from avalanche.models import SimpleMLP
    if torch.cuda.is_available():
        device = 'cuda:0'
        torch.backends.cudnn.benchmark = True
        device_count = torch.cuda.device_count()
        print(f"Found {device_count} CUDA GPU devices.")
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    print(f'Using {device} device')

     # --- BENCHMARK CREATION
    benchmark = SplitMNIST(n_experiences=10, seed=1234)
    # benchmark = CORe50(scenario="nc", mini=True, object_lvl=True)
    # ---------

    # MODEL CREATION
    # model = SimpleMLP(num_classes=50, input_size=32*32*3)
    # MODEL CREATION
    model = MlpVAE((3, 32, 32), nhid=2, device=device)

    # choose some metrics and evaluation method
    interactive_logger = InteractiveLogger()

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(
            minibatch=True, epoch=True, train_experience=True, stream=True
        ),
        loss_metrics(minibatch=True, epoch=True, train_experience=True, stream=True),
        forgetting_metrics(train_experience=True),
        loggers=[interactive_logger],
    )

    # CREATE THE STRATEGY INSTANCE (GenerativeReplay)
    cl_strategy = GenerativeReplay(
        model,
        torch.optim.Adam(model.parameters(), lr=0.001),
        CrossEntropyLoss(),
        train_mb_size=100,
        train_epochs=4,
        eval_mb_size=100,
        device=device,
        evaluator=eval_plugin,
    )

    # TRAINING LOOP
    print("Starting experiment...")
    results = []
    for train_experience in benchmark.train_stream:
        print("Start of train_experience ", train_experience.current_experience)
        cl_strategy.train(train_experience)
        print("Training completed")

        print("Computing accuracy on the whole test set")
        results.append(cl_strategy.eval(benchmark.test_stream))



def GR_Plugin_sandbox():

    if torch.cuda.is_available():
        device = 'cuda:0'
        torch.backends.cudnn.benchmark = True
        device_count = torch.cuda.device_count()
        print(f"Found {device_count} CUDA GPU devices.")
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    print(f'Using {device} device')


    # --- BENCHMARK CREATION
    benchmark = CORe50(scenario="nc", mini=True, object_lvl=True)
    # ---------

    # MODEL CREATION
    model = MlpVAE((3, 32, 32), nhid=2, device=device)

    eval_plugin = EvaluationPlugin(
    accuracy_metrics(
        minibatch=True, epoch=True, train_experience=True, stream=True
    ),
    loss_metrics(minibatch=True, epoch=True, train_experience=True, stream=True),
    forgetting_metrics(train_experience=True),
    )

    # CREATE THE STRATEGY INSTANCE (GenerativeReplay)
    cl_strategy = GenerativeReplay(
        model,
        torch.optim.Adam(model.parameters(), lr=0.001),
        CrossEntropyLoss(),
        train_mb_size=100,
        train_epochs=4,
        eval_mb_size=100,
        device=device,
        evaluator=eval_plugin,
    )

    # TRAINING LOOP
    print("Starting experiment...")
    f, axarr = plt.subplots(benchmark.n_experiences, 10)
    k = 0
    for train_experience in benchmark.train_stream:
        print("Start of train_experience ", train_experience.current_experience)
        cl_strategy.train(train_experience)
        print("Training completed")

        samples = model.generate(10)
        samples = samples.detach().cpu().numpy()

        for j in range(10):
            axarr[k, j].imshow(samples[j, 0], cmap="gray")
            axarr[k, 4].set_title("Generated images for train_experience " + str(k))
        np.vectorize(lambda ax: ax.axis("off"))(axarr)
        k += 1

    f.subplots_adjust(hspace=1.2)
    plt.savefig("VAE_output_per_exp")
    plt.show()


def train_split():

    # Load the CORe50 dataset
    core50 = CORe50(scenario="nc", mini=True, object_lvl=True)

    # the task label of each train_experience.
    print('--- Task labels:')
    print(core50.task_labels)

    # Instatiate train and test streams
    train_stream = core50.train_stream[0]
    test_stream = core50.test_stream


    print("Length", len(train_stream[0].dataset))



def wandb_import():
    api = wandb.Api() 
    run = api.run("continualearning/avalanche/l3fmb7v7") 

    # Get the history dictionary
    history = run.history()

    # Print the keys of the history dictionary
    print(history.keys())

    history = run.scan_history()
    GPU_W = [row["MaxGPU0Usage_Stream/eval_phase/test_stream/Task000"] for row in history]

    run_df = pd.DataFrame({"GPUW_W": GPU_W})

    run_df.to_csv("project1.csv")

if __name__ == "__main__":
    project_name = "New_Instances"
    # train_without_ES("Experiment_2")
    train_with_ES(project_name, 5)
    # train_iteratively_ES(project_name, 10)
    # cumulative_only(project_name, 5)
    # cumulative_via_plugin(project_name, 5)
    # extract_metrics(project_name)
    # extract_accuracy_valid(project_name)
    # extract_convergence(project_name, 10)
    # wandb_import()
    # GR_Plugin()
    # GR_GR()
    # GR_Plugin_sandbox()
    # test_tbplugin()
    # gem()
    # ewc()
    # train_split()

# [VAETraining(
#     model, optimizer, device=device,
#     train_mb_size=batchsize_train, train_epochs=epochs, eval_mb_size=batchsize_eval, evaluator=eval_plugin, 
#     plugins=[GenerativeReplayPlugin()]
# ),
