from avalanche.benchmarks.classic import CORe50
from avalanche.benchmarks.datasets import CORe50Dataset
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
import pandas as pd 
from avalanche.benchmarks.utils.utils import concat_datasets
from avalanche.benchmarks.generators import nc_benchmark
from torchvision import transforms
import wandb

from avalanche.benchmarks.generators import benchmark_with_validation_stream, class_balanced_split_strategy


PATIENCE = 25
EPOCHS = 44
N_CLASSES = 50


def transform():
    return transforms.Compose([
        # you can add other transformations in this list
        transforms.ToTensor()
    ])

def Generator_Strategy(n_classes, lr, batchsize_train, batchsize_eval, epochs, device, nhid, ES_plugin):

    # model:
    # First argument is shape of input sample so 3 for RGB 32x32 for image resolution
    generator = MlpVAE((3, 32, 32), nhid, n_classes, device=device)
    optimizer = Adam(generator.parameters(), lr=lr)
    # optimzer:

    # strategy (with plugin):
    generator_strategy = VAETraining(
        model=generator,
        optimizer=optimizer,
        train_mb_size=batchsize_train,
        train_epochs=epochs,
        eval_mb_size=batchsize_eval,
        device=device,
        plugins=[
            GenerativeReplayPlugin(),
            ES_plugin
        ],
    )

    return generator_strategy

def Generator_Strategy_without_ES(n_classes, lr, batchsize_train, batchsize_eval, epochs, device, nhid):

    # model:
    # First argument is shape of input sample so 3 for RGB 32x32 for image resolution
    generator = MlpVAE((3, 32, 32), nhid, n_classes, device=device)
    optimizer = Adam(generator.parameters(), lr=lr)
    # optimzer:

    # strategy (with plugin):
    generator_strategy = VAETraining(
        model=generator,
        optimizer=optimizer,
        train_mb_size=batchsize_train,
        train_epochs=epochs,
        eval_mb_size=batchsize_eval,
        device=device,
        plugins=[
            GenerativeReplayPlugin()
        ],
    )

    return generator_strategy


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
    batchsize_train = 512
    batchsize_eval = 512
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
            plugins=[EarlyStoppingPlugin(patience, "valid", ES_metric)], eval_every=eval_every)

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


def cumulative_vs_naive(project_name, num_runs):

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

    object_lvl=True

    if object_lvl:
        num_classes=50
    else:
        num_classes=10

    model = SimpleCNN(num_classes=num_classes).to(device)
    print(f"Main model {model}")

    # Load the CORe50 dataset
    core50 = CORe50(scenario="nc", mini=True, object_lvl=object_lvl)

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
    batchsize_train = 512
    batchsize_eval = 512
    patience = PATIENCE
    eval_every = 1


    for i in range(num_runs):

        loggers = []
        results = []
        loggers.append(WandBLogger(project_name=project_name, run_name="Naive", params={"reinit": True, "group": project_name}))
        loggers.append(InteractiveLogger())
        loggers.append(TextLogger(open(f'Naive_{i}.txt', 'a')))

        eval_plugin = EvaluationPlugin(
        loss_metrics(epoch=True, stream=True),
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
            train_mb_size=batchsize_train, train_epochs=epochs, eval_mb_size=batchsize_eval, evaluator=eval_plugin, \
            eval_every=eval_every, plugins=[EarlyStoppingPlugin(patience, "valid", "Loss_Stream", "min")])
        

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

        cl_strategy.train(train_experience, eval_streams=[val_stream])
        # results.append(cl_strategy.eval(val_experience))

        cl_strategy.eval(test_stream)

        loggers = []
        results = []
        loggers.append(WandBLogger(project_name=project_name, run_name="Cumulative", params={"reinit": True, "group": project_name}))
        loggers.append(InteractiveLogger())
        loggers.append(TextLogger(open(f'Cumulative_{i}.txt', 'a')))

        eval_plugin = EvaluationPlugin(
        loss_metrics(epoch=True, stream=True),
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


        cl_strategy = Cumulative(
            model, optimizer, criterion, device=device,
            train_mb_size=batchsize_train, train_epochs=epochs, eval_mb_size=batchsize_eval, evaluator=eval_plugin, \
            plugins=[EarlyStoppingPlugin(patience, "valid", "Loss_Stream", "min")], eval_every=eval_every)

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

        cl_strategy.train(train_experience, eval_streams=[val_stream])
        # results.append(cl_strategy.eval(val_experience))

        cl_strategy.eval(test_stream)

    # Finish the current WandB run
    wandb.finish()

def concat_cumulative_naive(project_name, num_runs):

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

    object_lvl=True

    if object_lvl:
        num_classes=50
    else:
        num_classes=10

    model = SimpleCNN(num_classes=num_classes).to(device)
    print(f"Main model {model}")

    # Load the CORe50 dataset
    core50 = CORe50(scenario="nc", mini=True, object_lvl=object_lvl)

    # the task label of each train_experience.
    print('--- Task labels:')
    print(core50.task_labels)

    # Instatiate train, validation and test streams
    validation_size = 0.2
    foo = lambda exp: class_balanced_split_strategy(validation_size, exp)
    core50 = benchmark_with_validation_stream(core50, custom_split_strategy=foo)

    model = SimpleCNN(num_classes=num_classes).to(device)

    print(f"Main model {model}")

    train_stream = core50.train_stream
    val_stream = core50.valid_stream
    test_stream = core50.test_stream


    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = CrossEntropyLoss()
    epochs = 10
    batchsize_train = 512
    batchsize_eval = 512
    patience = PATIENCE
    eval_every = 1


    for i in range(num_runs):

        loggers = []
        results = []
        loggers.append(WandBLogger(project_name=project_name, run_name="Naive", params={"reinit": True, "group": project_name}))
        loggers.append(InteractiveLogger())
        loggers.append(TextLogger(open(f'Naive_{i}.txt', 'a')))

        eval_plugin = EvaluationPlugin(
        loss_metrics(epoch=True, stream=True),
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
            train_mb_size=batchsize_train, train_epochs=epochs, eval_mb_size=batchsize_eval, evaluator=eval_plugin, \
            eval_every=eval_every, plugins=[EarlyStoppingPlugin(patience, "valid", "Loss_Stream", "min")])
        

        print(f"Current training strategy: {cl_strategy}")
    for i, (train_experience, val_experience) in enumerate(zip(train_stream, val_stream), 1):

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

        # if val_experience.current_experience == 0:
        #     concat_val = val_experience
            # else:
        #     concat_val = ConcatDataset([concat_val, val_experience])

        cl_strategy.train(train_experience, eval_streams=[val_stream[:i]])
        # results.append(cl_strategy.eval(val_experience))

        cl_strategy.eval(test_stream)

        loggers = []
        results = []
        loggers.append(WandBLogger(project_name=project_name, run_name="Cumulative", params={"reinit": True, "group": project_name}))
        loggers.append(InteractiveLogger())
        loggers.append(TextLogger(open(f'Cumulative_{i}.txt', 'a')))

        eval_plugin = EvaluationPlugin(
        loss_metrics(epoch=True, stream=True),
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


        cl_strategy = Cumulative(
            model, optimizer, criterion, device=device,
            train_mb_size=batchsize_train, train_epochs=epochs, eval_mb_size=batchsize_eval, evaluator=eval_plugin, \
            plugins=[EarlyStoppingPlugin(patience, "valid", "Loss_Stream", "min")], eval_every=eval_every)

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

        if val_experience.current_experience == 0:
            concat_val = val_experience
        else:
            concat_val = concat_datasets([concat_val, val_experience])

            cl_strategy.train(train_experience, eval_streams=[concat_val])
        # results.append(cl_strategy.eval(val_experience))

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

    object_lvl=False

    if object_lvl:
        num_classes=50
    else:
        num_classes=10

    model = SimpleCNN(num_classes=num_classes).to(device)
    print(f"Main model {model}")

    # Load the CORe50 dataset
    core50 = CORe50(scenario="nc", mini=True, object_lvl=object_lvl)

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
    batchsize_train = 512
    batchsize_eval = 512
    patience = PATIENCE + 20
    eval_every = 1


    for i in range(num_runs):

        loggers = []
        loggers.append(WandBLogger(project_name=project_name, run_name="Cumulative", params={"reinit": True, "group": group_name}))
        loggers.append(InteractiveLogger())
        loggers.append(TextLogger(open(f'Cumulative_{i}.txt', 'a')))

        eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=True,
            epoch=True,
            epoch_running=True,
            experience=True,
            stream=True),
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
            plugins=[EarlyStoppingPlugin(patience, "valid", ES_metric)], eval_every=eval_every)

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


def CWRStar_hiperparameter_search(project_name, ES_metric, min_max, scenario):

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

    # Load the CORe50 dataset
    if scenario == "ni":
        core50 = CORe50(scenario="ni", mini=True, object_lvl=True)
    elif scenario == "nc":
        # Load the CORe50 dataset
        core50_train = CORe50Dataset(train=True, mini=True)
        core50_test = CORe50Dataset(train=False, mini=True)
        # Create different split
        core50 = nc_benchmark(core50_train, core50_test, n_experiences=10, shuffle=False, task_labels=False, train_transform=transform(), eval_transform=transform())
    else:
        raise NameError("Scenario name unknown")

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


    criterion = CrossEntropyLoss()
    epochs = 50
    batchsize_train = 128
    batchsize_eval = 128
    patience = PATIENCE
    eval_every = 2
    lr = 0.001

    layers = ['classifier.0', 'features.12', 'features.6', 'features.0']

    for layer in layers:

        loggers = []
        loggers.append(WandBLogger(project_name=project_name, run_name=f"CWRStar_layer_{layer}", params={"reinit": True, "group": "CWRStar_group"}))
        loggers.append(InteractiveLogger())
        # loggers.append(TensorboardLogger(filename_suffix=f"EWC_lambda_{ewc_lambda}"))
        loggers.append(TextLogger(open(f"CWRStar_layer_{layer}.txt", 'a')))

        eval_plugin = EvaluationPlugin(
        loss_metrics(epoch=True, stream=True),
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

        model = SimpleCNN(num_classes=50).to(device)
        print(f"Main model {model}")
        print("Model layers names\n")
        for name, param in model.named_parameters():
            print(name)
        print("\n")
        optimizer = Adam(model.parameters(), lr=lr)


        cl_strategy = CWRStar(
            model, optimizer, criterion, cwr_layer_name=layer, device=device,
            train_mb_size=batchsize_train, train_epochs=epochs, eval_mb_size=batchsize_eval, evaluator=eval_plugin, \
            plugins=[EarlyStoppingPlugin(patience, "valid", ES_metric, min_max)], eval_every=eval_every)

        print(f"Current training strategy: {cl_strategy}")
        for i, train_experience in enumerate(train_stream, 1):
            # Accumulate the experiences
            val_experience = val_stream[:i]
            print(f"Experience number train {train_experience.current_experience}")
            print(f"classes in this experience train {train_experience.classes_in_this_experience}")
            print(f"Classes seen so far train {train_experience.classes_seen_so_far}")
            print(f"Training on train {len(train_experience.dataset)} examples")

            cl_strategy.train(train_experience, eval_streams=[val_experience])

            cl_strategy.eval(test_stream)


    # Finish the current WandB run
    # wandb.finish()

def GEM_hiperparemeter_search(project_name, ES_metric, min_max, scenario):

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

    # Load the CORe50 dataset
    if scenario == "ni":
        core50 = CORe50(scenario="ni", mini=True, object_lvl=True)
    elif scenario == "nc":
        # Load the CORe50 dataset
        core50_train = CORe50Dataset(train=True, mini=True)
        core50_test = CORe50Dataset(train=False, mini=True)
        # Create different split
        core50 = nc_benchmark(core50_train, core50_test, n_experiences=10, shuffle=False, task_labels=False, train_transform=transform(), eval_transform=transform())
    else:
        raise NameError("Scenario name unknown")

    # Instatiate train, validation and test streams
    validation_size = 0.2
    foo = lambda exp: class_balanced_split_strategy(validation_size, exp)
    core50 = benchmark_with_validation_stream(core50, custom_split_strategy=foo)

    train_stream = core50.train_stream
    val_stream = core50.valid_stream
    test_stream = core50.test_stream

    criterion = CrossEntropyLoss()
    epochs = 50
    batchsize_train = 128
    batchsize_eval = 128
    patience = PATIENCE
    eval_every = 2
    lr = 0.001

    # How many images are in the memory
    patterns_per_experience = [64, 1024, 8192]
    memory_strength = [0.01, 1, 100]

    for ppe in patterns_per_experience:
        for ms in memory_strength:

            loggers = []
            loggers.append(WandBLogger(project_name=project_name, run_name=f"GEM_ms_{ms}_ppe_{ppe}", params={"reinit": True, "group": "GEM_group"}))
            loggers.append(InteractiveLogger())
            # loggers.append(TensorboardLogger(filename_suffix=f'GEM_ms_{ms}_ppe_{ppe}'))
            loggers.append(TextLogger(open(f'GEM_ms_{ms}_ppe_{ppe}.txt', 'a')))

            eval_plugin = EvaluationPlugin(
            loss_metrics(epoch=True, stream=True),
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

            model = SimpleCNN(num_classes=50).to(device)
            print(f"Main model {model}")
            optimizer = Adam(model.parameters(), lr=lr)

            cl_strategy = GEM(
                model, optimizer, criterion, device=device, patterns_per_exp=ppe, memory_strength=ms,
                train_mb_size=batchsize_train, train_epochs=epochs, eval_mb_size=batchsize_eval, evaluator=eval_plugin,
                plugins=[EarlyStoppingPlugin(patience, "valid", ES_metric, min_max)], eval_every=eval_every)

            print(f"Current training strategy: {cl_strategy}")
            for i, train_experience in enumerate(train_stream, 1):
                # Accumulate the experiences
                val_experience = val_stream[:i]
                print(f"Experience number train {train_experience.current_experience}")
                print(f"classes in this experience train {train_experience.classes_in_this_experience}")
                print(f"Classes seen so far train {train_experience.classes_seen_so_far}")
                print(f"Training on train {len(train_experience.dataset)} examples")

                cl_strategy.train(train_experience, eval_streams=[val_experience])

                cl_strategy.eval(test_stream)

    # Finish the current WandB run
    # wandb.finish()
 

def EWC_hiperparameter_search(project_name, ES_metric, scenario):

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

    # Load the CORe50 dataset
    if scenario == "ni":
        core50 = CORe50(scenario="ni", mini=True, object_lvl=True)
    elif scenario == "nc":
        # Load the CORe50 dataset
        core50_train = CORe50Dataset(train=True, mini=True)
        core50_test = CORe50Dataset(train=False, mini=True)
        # Create different split
        core50 = nc_benchmark(core50_train, core50_test, n_experiences=10, shuffle=False, task_labels=False, train_transform=transform(), eval_transform=transform())
    else:
        raise NameError("Scenario name unknown")

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


    criterion = CrossEntropyLoss()
    epochs = 50
    batchsize_train = 128
    batchsize_eval = 128
    patience = PATIENCE
    eval_every = 2
    lr = 0.001

    if ES_metric == "Top1_Acc_Stream":
        min_max = "max"
    elif ES_metric == "Loss_Stream":
        min_max = "min"
    else:
        raise NameError

    ewc_lambdas = [0.1, 1, 10, 100]

    for ewc_lambda in ewc_lambdas:

        loggers = []
        loggers.append(WandBLogger(project_name=project_name, run_name=f"EWC_lambda_{ewc_lambda}", params={"reinit": True, "group": "EWC_group"}))
        loggers.append(InteractiveLogger())
        # loggers.append(TensorboardLogger(filename_suffix=f"EWC_lambda_{ewc_lambda}"))
        loggers.append(TextLogger(open(f"EWC_lambda_{ewc_lambda}.txt", 'a')))

        eval_plugin = EvaluationPlugin(
        loss_metrics(epoch=True, stream=True),
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

        model = SimpleCNN(num_classes=50).to(device)
        print(f"Main model {model}")
        optimizer = Adam(model.parameters(), lr=lr)


        cl_strategy = EWC(
            model, optimizer, criterion, ewc_lambda=ewc_lambda, device=device,
            train_mb_size=batchsize_train, train_epochs=epochs, eval_mb_size=batchsize_eval, evaluator=eval_plugin, \
            plugins=[EarlyStoppingPlugin(patience, "valid", ES_metric, min_max)], eval_every=eval_every)

        print(f"Current training strategy: {cl_strategy}")
        for i, train_experience in enumerate(train_stream, 1):
            # Accumulate the experiences
            val_experience = val_stream[:i]
            print(f"Experience number train {train_experience.current_experience}")
            print(f"classes in this experience train {train_experience.classes_in_this_experience}")
            print(f"Classes seen so far train {train_experience.classes_seen_so_far}")
            print(f"Training on train {len(train_experience.dataset)} examples")

            cl_strategy.train(train_experience, eval_streams=[val_experience])

            cl_strategy.eval(test_stream)


    # Finish the current WandB run
    # wandb.finish()



def GR_hiperparameter_search(project_name, ES_metric, min_max, scenario):

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

    # Load the CORe50 dataset
    if scenario == "ni":
        core50 = CORe50(scenario="ni", mini=True, object_lvl=True)
    elif scenario == "nc":
        # Load the CORe50 dataset
        core50_train = CORe50Dataset(train=True, mini=True)
        core50_test = CORe50Dataset(train=False, mini=True)
        # Create different split
        core50 = nc_benchmark(core50_train, core50_test, n_experiences=10, shuffle=False, task_labels=False, train_transform=transform(), eval_transform=transform())
    else:
        raise NameError("Scenario name unknown")

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


    criterion = CrossEntropyLoss()
    epochs = 50
    batchsize_train = 128
    batchsize_eval = 128
    patience = PATIENCE
    eval_every = 2
    lr = 0.001

    nhids = [2, 10, 25, 50]

    for nhid in nhids:

        generator_strategy = Generator_Strategy(N_CLASSES, lr, batchsize_train, batchsize_eval,\
                epochs, device, nhid, EarlyStoppingPlugin(patience, "valid", ES_metric, min_max))
        
        loggers = []
        loggers.append(WandBLogger(project_name=project_name, run_name="GenerativeReplay", params={"reinit": True, "group": project_name}))
        loggers.append(InteractiveLogger())
        # loggers.append(TensorboardLogger(filename_suffix=f"GR_{nhids}"))
        loggers.append(TextLogger(open(f'GenerativeReplay_{nhids}.txt', 'a')))

        eval_plugin = EvaluationPlugin(
        loss_metrics(epoch=True, stream=True),
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

        model = SimpleCNN(num_classes=N_CLASSES).to(device)
        print(f"Main model {model}")
        optimizer = Adam(model.parameters(), lr=lr)

        cl_strategy = GenerativeReplay(
            model, optimizer, criterion, device=device,
            train_mb_size=batchsize_train, train_epochs=epochs, eval_mb_size=batchsize_eval, evaluator=eval_plugin, \
            plugins=[EarlyStoppingPlugin(patience, "valid", ES_metric, min_max)], eval_every=eval_every, generator_strategy=generator_strategy)

        print(f"Current training strategy: {cl_strategy}")
        for i, train_experience in enumerate(train_stream, 1):
            # Accumulate the experiences
            val_experience = val_stream[:i]
            print(f"Experience number train {train_experience.current_experience}")
            print(f"classes in this experience train {train_experience.classes_in_this_experience}")
            print(f"Classes seen so far train {train_experience.classes_seen_so_far}")
            print(f"Training on train {len(train_experience.dataset)} examples")

            cl_strategy.train(train_experience, eval_streams=[val_experience])

            cl_strategy.eval(test_stream)



    # Finish the current WandB run
    # wandb.finish()




def core50_check():

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

    object_lvl=True

    if object_lvl:
        num_classes=50
    else:
        num_classes=10

    model = SimpleCNN(num_classes=num_classes).to(device)
    print(f"Main model {model}")

    # Load the CORe50 dataset
    core50_train = CORe50Dataset(train=True, mini=True)
    core50_test = CORe50Dataset(train=False, mini=True)

    # # the task label of each train_experience.
    # print('--- Task labels:')
    # print(core50.task_labels)

    # Create different split
    core50 = nc_benchmark(core50_train, core50_test, n_experiences=10, shuffle=False, task_labels=False, train_transform=transform(), eval_transform=transform())


    # Instatiate train, validation and test streams
    validation_size = 0.2
    foo = lambda exp: class_balanced_split_strategy(validation_size, exp)
    core50 = benchmark_with_validation_stream(core50, custom_split_strategy=foo)

    train_stream = core50.train_stream
    val_stream = core50.valid_stream
    test_stream = core50.test_stream

    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = CrossEntropyLoss()
    epochs = 1
    batchsize_train = 512
    batchsize_eval = 512
    patience = PATIENCE
    eval_every = 2

    loggers = []
    loggers.append(InteractiveLogger())
    
    eval_plugin = EvaluationPlugin(
    loss_metrics(epoch=True, stream=True),
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

    cl_strategy = EWC(
        model, optimizer, criterion, ewc_lambda=1, device=device,
        train_mb_size=batchsize_train, train_epochs=epochs, eval_mb_size=batchsize_eval, evaluator=eval_plugin, \
        plugins=[EarlyStoppingPlugin(patience, "valid", "Top1_Acc_Stream", "max")], eval_every=eval_every)

    print(f"Current training strategy: {cl_strategy}")
    for i, train_experience in enumerate(train_stream, 1):
        # Accumulate the experiences
        val_experience = val_stream[:i]
        print(f"Experience number train {train_experience.current_experience}")
        print(f"classes in this experience train {train_experience.classes_in_this_experience}")
        print(f"Classes seen so far train {train_experience.classes_seen_so_far}")
        print(f"Training on train {len(train_experience.dataset)} examples")
        # train_experience = ToTensor()(train_experience)

        cl_strategy.train(train_experience, eval_streams=[val_experience])

    cl_strategy.eval(test_stream)


def train_with_ES(group_name, num_runs, ES_metric, scenario):

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

    # Load the CORe50 dataset
    if scenario == "ni":
        core50 = CORe50(scenario="ni", mini=True, object_lvl=True)
    elif scenario == "nc":
        # Load the CORe50 dataset
        core50_train = CORe50Dataset(train=True, mini=True)
        core50_test = CORe50Dataset(train=False, mini=True)
        # Create different split
        core50 = nc_benchmark(core50_train, core50_test, n_experiences=10, shuffle=False, task_labels=False, train_transform=transform(), eval_transform=transform())
    else:
        raise NameError("Scenario name unknown")

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

    criterion = CrossEntropyLoss()
    lr = 0.001
    epochs = 50
    batchsize_train = 128
    batchsize_eval = 128
    patience = PATIENCE
    eval_every = 2
    if ES_metric == "Top1_Acc_Stream":
        min_max = "max"
    elif ES_metric == "Loss_Stream":
        min_max = "min"
    else:
        raise NameError

    generator_strategy = Generator_Strategy(N_CLASSES, lr, batchsize_train, batchsize_eval,\
                    epochs, device, 2, EarlyStoppingPlugin(patience, "valid", ES_metric, min_max))

    for j in range(num_runs):

        loggers = []
        loggers.append(WandBLogger(project_name=project_name, run_name="Naive", params={"reinit": True, "group": group_name}))
        loggers.append(InteractiveLogger())
        loggers.append(TextLogger(open(f'Naive_{j}.txt', 'a')))

        eval_plugin = EvaluationPlugin(
        loss_metrics(epoch=True, stream=True),
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

        model = SimpleCNN(num_classes=50).to(device)
        print(f"Main model {model}")
        optimizer = Adam(model.parameters(), lr=lr)

        cl_strategy = Naive(
            model, optimizer, criterion, device=device,
            train_mb_size=batchsize_train, train_epochs=epochs, eval_mb_size=batchsize_eval, evaluator=eval_plugin, \
            eval_every=eval_every, plugins=[EarlyStoppingPlugin(patience, "valid", ES_metric, min_max)])
        

        print(f"Current training strategy: {cl_strategy}")
        for i, train_experience in enumerate(train_stream, 1):
            # Accumulate the experiences
            val_experience = val_stream[:i]
            print(f"Experience number train {train_experience.current_experience}")
            print(f"classes in this experience train {train_experience.classes_in_this_experience}")
            print(f"Classes seen so far train {train_experience.classes_seen_so_far}")
            print(f"Training on train {len(train_experience.dataset)} examples")

            cl_strategy.train(train_experience, eval_streams=[val_experience])

            cl_strategy.eval(test_stream)


        loggers = []
        loggers.append(WandBLogger(project_name=project_name, run_name="CWR*", params={"reinit": True, "group": group_name}))
        loggers.append(InteractiveLogger())
        loggers.append(TextLogger(open(f'CWR*_{j}.txt', 'a')))

        eval_plugin = EvaluationPlugin(
        loss_metrics(epoch=True, stream=True),
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

        model = SimpleCNN(num_classes=50).to(device)
        print(f"Main model {model}")
        optimizer = Adam(model.parameters(), lr=lr)
        
        cl_strategy = CWRStar(
            model, optimizer, criterion, cwr_layer_name='classifier.0', device=device,
            train_mb_size=batchsize_train, train_epochs=epochs, eval_mb_size=batchsize_eval, evaluator=eval_plugin, \
            plugins=[EarlyStoppingPlugin(patience, "valid", ES_metric, min_max)], eval_every=eval_every)

        print(f"Current training strategy: {cl_strategy}")
        for i, train_experience in enumerate(train_stream, 1):
            # Accumulate the experiences
            val_experience = val_stream[:i]
            print(f"Experience number train {train_experience.current_experience}")
            print(f"classes in this experience train {train_experience.classes_in_this_experience}")
            print(f"Classes seen so far train {train_experience.classes_seen_so_far}")
            print(f"Training on train {len(train_experience.dataset)} examples")

            cl_strategy.train(train_experience, eval_streams=[val_experience])

            cl_strategy.eval(test_stream)


        loggers = []
        loggers.append(WandBLogger(project_name=project_name, run_name="GEM", params={"reinit": True, "group": group_name}))
        loggers.append(InteractiveLogger())
        loggers.append(TextLogger(open(f'GEM_{j}.txt', 'a')))

        eval_plugin = EvaluationPlugin(
        loss_metrics(epoch=True, stream=True),
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

        model = SimpleCNN(num_classes=50).to(device)
        print(f"Main model {model}")
        optimizer = Adam(model.parameters(), lr=lr)

        cl_strategy = GEM(
            model, optimizer, criterion, device=device, patterns_per_exp=1024, memory_strength=1,
            train_mb_size=batchsize_train, train_epochs=epochs, eval_mb_size=batchsize_eval, evaluator=eval_plugin,
            plugins=[EarlyStoppingPlugin(patience, "valid", ES_metric, min_max)], eval_every=eval_every)

        print(f"Current training strategy: {cl_strategy}")
        for i, train_experience in enumerate(train_stream, 1):
            # Accumulate the experiences
            val_experience = val_stream[:i]
            print(f"Experience number train {train_experience.current_experience}")
            print(f"classes in this experience train {train_experience.classes_in_this_experience}")
            print(f"Classes seen so far train {train_experience.classes_seen_so_far}")
            print(f"Training on train {len(train_experience.dataset)} examples")

            cl_strategy.train(train_experience, eval_streams=[val_experience])

            cl_strategy.eval(test_stream)
        
        loggers = []
        loggers.append(WandBLogger(project_name=project_name, run_name="EWC", params={"reinit": True, "group": group_name}))
        loggers.append(InteractiveLogger())
        loggers.append(TextLogger(open(f'EWC_{j}.txt', 'a')))

        eval_plugin = EvaluationPlugin(
        loss_metrics(epoch=True, stream=True),
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

        model = SimpleCNN(num_classes=50).to(device)
        print(f"Main model {model}")
        optimizer = Adam(model.parameters(), lr=lr)

        cl_strategy = EWC(
            model, optimizer, criterion, ewc_lambda=0.1, device=device,
            train_mb_size=batchsize_train, train_epochs=epochs, eval_mb_size=batchsize_eval, evaluator=eval_plugin, \
            plugins=[EarlyStoppingPlugin(patience, "valid", ES_metric, min_max)], eval_every=eval_every)

        print(f"Current training strategy: {cl_strategy}")
        for i, train_experience in enumerate(train_stream, 1):
            # Accumulate the experiences
            val_experience = val_stream[:i]
            print(f"Experience number train {train_experience.current_experience}")
            print(f"classes in this experience train {train_experience.classes_in_this_experience}")
            print(f"Classes seen so far train {train_experience.classes_seen_so_far}")
            print(f"Training on train {len(train_experience.dataset)} examples")

            cl_strategy.train(train_experience, eval_streams=[val_experience])

            cl_strategy.eval(test_stream)


        loggers = []
        loggers.append(WandBLogger(project_name=project_name, run_name="GenerativeReplay", params={"reinit": True, "group": group_name}))
        loggers.append(InteractiveLogger())
        loggers.append(TextLogger(open(f'GenerativeReplay_{j}.txt', 'a')))

        eval_plugin = EvaluationPlugin(
        loss_metrics(epoch=True, stream=True),
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

        model = SimpleCNN(num_classes=50).to(device)
        print(f"Main model {model}")
        optimizer = Adam(model.parameters(), lr=lr)

        cl_strategy = GenerativeReplay(
            model, optimizer, criterion, device=device,
            train_mb_size=batchsize_train, train_epochs=epochs, eval_mb_size=batchsize_eval, evaluator=eval_plugin, \
            plugins=[EarlyStoppingPlugin(patience, "valid", ES_metric, min_max)], eval_every=eval_every, generator_strategy=generator_strategy)

        print(f"Current training strategy: {cl_strategy}")
        for i, train_experience in enumerate(train_stream, 1):
            # Accumulate the experiences
            val_experience = val_stream[:i]
            print(f"Experience number train {train_experience.current_experience}")
            print(f"classes in this experience train {train_experience.classes_in_this_experience}")
            print(f"Classes seen so far train {train_experience.classes_seen_so_far}")
            print(f"Training on train {len(train_experience.dataset)} examples")

            cl_strategy.train(train_experience, eval_streams=[val_experience])

            cl_strategy.eval(test_stream)


        loggers = []
        loggers.append(WandBLogger(project_name=project_name, run_name="Cumulative", params={"reinit": True, "group": group_name}))
        loggers.append(InteractiveLogger())
        loggers.append(TextLogger(open(f'Cumulative_{j}.txt', 'a')))

        eval_plugin = EvaluationPlugin(
        loss_metrics(epoch=True, stream=True),
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

        model = SimpleCNN(num_classes=50).to(device)
        print(f"Main model {model}")
        optimizer = Adam(model.parameters(), lr=lr)

        cl_strategy = Cumulative(
            model, optimizer, criterion, device=device,
            train_mb_size=batchsize_train, train_epochs=epochs, eval_mb_size=batchsize_eval, evaluator=eval_plugin, \
            plugins=[EarlyStoppingPlugin(patience, "valid", ES_metric, min_max)], eval_every=eval_every)

        print(f"Current training strategy: {cl_strategy}")
        for i, train_experience in enumerate(train_stream, 1):
            # Accumulate the experiences
            val_experience = val_stream[:i]
            print(f"Experience number train {train_experience.current_experience}")
            print(f"classes in this experience train {train_experience.classes_in_this_experience}")
            print(f"Classes seen so far train {train_experience.classes_seen_so_far}")
            print(f"Training on train {len(train_experience.dataset)} examples")

            cl_strategy.train(train_experience, eval_streams=[val_experience])

            cl_strategy.eval(test_stream)

    # Finish the current WandB run
    wandb.finish()



def train_with_ES_evaluation(group_name, num_runs, ES_metric, scenario):
    '''
    The same evaluation set experience as the train set
    '''
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

    # Load the CORe50 dataset
    if scenario == "ni":
        core50 = CORe50(scenario="ni", mini=True, object_lvl=True)
    elif scenario == "nc":
        # Load the CORe50 dataset
        core50_train = CORe50Dataset(train=True, mini=True)
        core50_test = CORe50Dataset(train=False, mini=True)
        # Create different split
        core50 = nc_benchmark(core50_train, core50_test, n_experiences=10, shuffle=False, task_labels=False, train_transform=transform(), eval_transform=transform())
    else:
        raise NameError("Scenario name unknown")

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

    criterion = CrossEntropyLoss()
    lr = 0.001
    epochs = EPOCHS
    batchsize_train = 128
    batchsize_eval = 128
    patience = PATIENCE
    eval_every = 2
    if ES_metric == "Top1_Acc_Stream":
        min_max = "max"
    elif ES_metric == "Loss_Stream":
        min_max = "min"
    else:
        raise NameError

    generator_strategy = Generator_Strategy(N_CLASSES, lr, batchsize_train, batchsize_eval,\
                    epochs, device, 2, EarlyStoppingPlugin(patience, "valid", ES_metric, min_max))

    for j in range(num_runs):

        loggers = []
        loggers.append(WandBLogger(project_name=project_name, run_name="Naive", params={"reinit": True, "group": group_name}))
        loggers.append(InteractiveLogger())
        loggers.append(TextLogger(open(f'Naive_{j}.txt', 'a')))

        eval_plugin = EvaluationPlugin(
        loss_metrics(epoch=True, stream=True),
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

        model = SimpleCNN(num_classes=50).to(device)
        print(f"Main model {model}")
        optimizer = Adam(model.parameters(), lr=lr)

        cl_strategy = Naive(
            model, optimizer, criterion, device=device,
            train_mb_size=batchsize_train, train_epochs=epochs, eval_mb_size=batchsize_eval, evaluator=eval_plugin, \
            eval_every=eval_every, plugins=[EarlyStoppingPlugin(patience, "valid", ES_metric, min_max)])
        

        print(f"Current training strategy: {cl_strategy}")
        for train_experience, val_experience in zip(train_stream, val_stream):
            print(f"Experience number train {train_experience.current_experience}")
            print(f"classes in this experience train {train_experience.classes_in_this_experience}")
            print(f"Classes seen so far train {train_experience.classes_seen_so_far}")
            print(f"Training on train {len(train_experience.dataset)} examples")

            cl_strategy.train(train_experience, eval_streams=[val_experience])

            cl_strategy.eval(test_stream)


        loggers = []
        loggers.append(WandBLogger(project_name=project_name, run_name="CWR*", params={"reinit": True, "group": group_name}))
        loggers.append(InteractiveLogger())
        loggers.append(TextLogger(open(f'CWR*_{j}.txt', 'a')))

        eval_plugin = EvaluationPlugin(
        loss_metrics(epoch=True, stream=True),
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


        model = SimpleCNN(num_classes=50).to(device)
        print(f"Main model {model}")
        optimizer = Adam(model.parameters(), lr=lr)
        
        cl_strategy = CWRStar(
            model, optimizer, criterion, cwr_layer_name='classifier.0', device=device,
            train_mb_size=batchsize_train, train_epochs=epochs, eval_mb_size=batchsize_eval, evaluator=eval_plugin, \
            plugins=[EarlyStoppingPlugin(patience, "valid", ES_metric, min_max)], eval_every=eval_every)

        print(f"Current training strategy: {cl_strategy}")
        for train_experience, val_experience in zip(train_stream, val_stream):
            print(f"Experience number train {train_experience.current_experience}")
            print(f"classes in this experience train {train_experience.classes_in_this_experience}")
            print(f"Classes seen so far train {train_experience.classes_seen_so_far}")
            print(f"Training on train {len(train_experience.dataset)} examples")

            cl_strategy.train(train_experience, eval_streams=[val_experience])

            cl_strategy.eval(test_stream)


        loggers = []
        loggers.append(WandBLogger(project_name=project_name, run_name="GEM", params={"reinit": True, "group": group_name}))
        loggers.append(InteractiveLogger())
        loggers.append(TextLogger(open(f'GEM_{j}.txt', 'a')))

        eval_plugin = EvaluationPlugin(
        loss_metrics(epoch=True, stream=True),
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


        model = SimpleCNN(num_classes=50).to(device)
        print(f"Main model {model}")
        optimizer = Adam(model.parameters(), lr=lr)

        cl_strategy = GEM(
            model, optimizer, criterion, device=device, patterns_per_exp=1024, memory_strength=1,
            train_mb_size=batchsize_train, train_epochs=epochs, eval_mb_size=batchsize_eval, evaluator=eval_plugin,
            plugins=[EarlyStoppingPlugin(patience, "valid", ES_metric, min_max)], eval_every=eval_every)

        print(f"Current training strategy: {cl_strategy}")
        for train_experience, val_experience in zip(train_stream, val_stream):
            print(f"Experience number train {train_experience.current_experience}")
            print(f"classes in this experience train {train_experience.classes_in_this_experience}")
            print(f"Classes seen so far train {train_experience.classes_seen_so_far}")
            print(f"Training on train {len(train_experience.dataset)} examples")

            cl_strategy.train(train_experience, eval_streams=[val_experience])

            cl_strategy.eval(test_stream)
        
        loggers = []
        loggers.append(WandBLogger(project_name=project_name, run_name="EWC", params={"reinit": True, "group": group_name}))
        loggers.append(InteractiveLogger())
        loggers.append(TextLogger(open(f'EWC_{j}.txt', 'a')))

        eval_plugin = EvaluationPlugin(
        loss_metrics(epoch=True, stream=True),
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

        model = SimpleCNN(num_classes=50).to(device)
        print(f"Main model {model}")
        optimizer = Adam(model.parameters(), lr=lr)

        cl_strategy = EWC(
            model, optimizer, criterion, ewc_lambda=100, device=device,
            train_mb_size=batchsize_train, train_epochs=epochs, eval_mb_size=batchsize_eval, evaluator=eval_plugin, \
            plugins=[EarlyStoppingPlugin(patience, "valid", ES_metric, min_max)], eval_every=eval_every)

        print(f"Current training strategy: {cl_strategy}")
        for train_experience, val_experience in zip(train_stream, val_stream):
            print(f"Experience number train {train_experience.current_experience}")
            print(f"classes in this experience train {train_experience.classes_in_this_experience}")
            print(f"Classes seen so far train {train_experience.classes_seen_so_far}")
            print(f"Training on train {len(train_experience.dataset)} examples")

            cl_strategy.train(train_experience, eval_streams=[val_experience])

            cl_strategy.eval(test_stream)


        loggers = []
        loggers.append(WandBLogger(project_name=project_name, run_name="GenerativeReplay", params={"reinit": True, "group": group_name}))
        loggers.append(InteractiveLogger())
        loggers.append(TextLogger(open(f'GenerativeReplay_{j}.txt', 'a')))

        eval_plugin = EvaluationPlugin(
        loss_metrics(epoch=True, stream=True),
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


        model = SimpleCNN(num_classes=50).to(device)
        print(f"Main model {model}")
        optimizer = Adam(model.parameters(), lr=lr)

        cl_strategy = GenerativeReplay(
            model, optimizer, criterion, device=device,
            train_mb_size=batchsize_train, train_epochs=epochs, eval_mb_size=batchsize_eval, evaluator=eval_plugin, \
            plugins=[EarlyStoppingPlugin(patience, "valid", ES_metric, min_max)], eval_every=eval_every, generator_strategy=generator_strategy)

        print(f"Current training strategy: {cl_strategy}")
        for train_experience, val_experience in zip(train_stream, val_stream):
            print(f"Experience number train {train_experience.current_experience}")
            print(f"classes in this experience train {train_experience.classes_in_this_experience}")
            print(f"Classes seen so far train {train_experience.classes_seen_so_far}")
            print(f"Training on train {len(train_experience.dataset)} examples")

            cl_strategy.train(train_experience, eval_streams=[val_experience])

            cl_strategy.eval(test_stream)


        loggers = []
        loggers.append(WandBLogger(project_name=project_name, run_name="Cumulative", params={"reinit": True, "group": group_name}))
        loggers.append(InteractiveLogger())
        loggers.append(TextLogger(open(f'Cumulative_{j}.txt', 'a')))

        eval_plugin = EvaluationPlugin(
        loss_metrics(epoch=True, stream=True),
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


        model = SimpleCNN(num_classes=50).to(device)
        print(f"Main model {model}")
        optimizer = Adam(model.parameters(), lr=lr)

        cl_strategy = Cumulative(
            model, optimizer, criterion, device=device,
            train_mb_size=batchsize_train, train_epochs=epochs, eval_mb_size=batchsize_eval, evaluator=eval_plugin, \
            plugins=[EarlyStoppingPlugin(patience, "valid", ES_metric, min_max)], eval_every=eval_every)

        print(f"Current training strategy: {cl_strategy}")
        for train_experience, val_experience in zip(train_stream, val_stream):
            print(f"Experience number train {train_experience.current_experience}")
            print(f"classes in this experience train {train_experience.classes_in_this_experience}")
            print(f"Classes seen so far train {train_experience.classes_seen_so_far}")
            print(f"Training on train {len(train_experience.dataset)} examples")

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
    batchsize_train = 512
    batchsize_eval = 512
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
    batchsize_train = 256
    batchsize_eval = 256
    patience = PATIENCE
    eval_every = 1
    lr = 0.005

    # Define list of strategies
    strategies = ['Naive', 'CWRStar', 'GEM', 'EWC', 'Cumulative']  # Add the names of other strategies here.

    # Iterate through different strategies
    for run_idx in range(num_runs):
        for strategy_name in strategies:
            # Reset the model and optimizer for each strategy
            model = SimpleCNN(num_classes=50).to(device)
            optimizer = Adam(model.parameters(), lr=lr)
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



def extract_metrics(project_name, metric, description):
    # Login to the wandb
    wandb.login()
    # Extract the metrics from WandB after all the runs
    api = wandb.Api()
    runs = api.runs(project_name)
    # Initialize empty DataFrame
    data = pd.DataFrame()

    # Fetch the logged metrics for each run
    for run in runs:
        history = run.history() # 'default or system'
        history['run_id'] = run.id
        history['strategy_name'] = run.name
        data = pd.concat([data, history], ignore_index=True)

    data.to_excel("metrics.xlsx", index=False)

    # Calculate mean gpu_usage per runtime for each strategy
    data.interpolate(inplace=True)

    # Create a helper column to detect changes in strategy_name
    data['index_run'] = data.groupby("run_id").cumcount()
    data[metric] = data[metric]*100

    df_mean = data.groupby(["strategy_name", "index_run"])[metric, "_step"].mean().reset_index()
    df_std = data.groupby(["strategy_name", "index_run"])[metric].std().reset_index()
    df_std['_step'] = data.groupby(["strategy_name", "index_run"])["_step"].mean().reset_index()["_step"]

    # Pivot data to have strategies as columns
    pivot_table = df_mean.pivot(index='_step', columns='strategy_name', values=metric)
    pivot_table_std = df_std.pivot(index='_step', columns='strategy_name', values=metric)

    pivot_table.interpolate(method='akima', inplace=True, limit=10)
    pivot_table_std.interpolate(method='akima', inplace=True, limit=10)

    plt.figure(figsize=(12, 6))

    # Define colors for each strategy
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    # Loop through each strategy
    for i, strategy in enumerate(pivot_table.columns):
        means = pivot_table[strategy] 
        stds = pivot_table_std[strategy] 
        x = pivot_table.index 
        # Plot means for this strategy with customizations
        plt.plot(x, means,
                label=strategy,
                linewidth=2,
                color=colors[i % len(colors)])
        
        # Add variance as shadowed region
        plt.fill_between(x, means - stds, means + stds,
                        color=colors[i % len(colors)], alpha=0.2)

    # Adding labels and title
    plt.xlabel('iterations')
    plt.ylabel(f'Mean {description}')
    plt.title(f'Mean {description} with Standard Deviation for Different Strategies')
    plt.legend()

    # Show plot
    plt.show()

def extract_system_metrics(project_name, metric, description):
    # Login to the wandb
    wandb.login()
    # Extract the metrics from WandB after all the runs
    api = wandb.Api()
    runs = api.runs(project_name)
    # Initialize empty DataFrame
    data = pd.DataFrame()

    # Fetch the logged metrics for each run
    for run in runs:
        history = run.history(stream="system") # 'default or system'
        history['run_id'] = run.id
        history['strategy_name'] = run.name
        data = pd.concat([data, history], ignore_index=True)

    data.to_excel("system_metrics.xlsx", index=False)

    # Calculate mean gpu_usage per runtime for each strategy
    data.interpolate(inplace=True)


    # Create a helper column to detect changes in strategy_name
    data['index_run'] = data.groupby("run_id").cumcount()

    df_mean= data.groupby(["strategy_name", "index_run"])[metric, "_runtime"].mean().reset_index()
    df_std = data.groupby(["strategy_name", "index_run"])[metric].std().reset_index()
    df_std['_step'] = data.groupby(["strategy_name", "index_run"])["_step"].mean().reset_index()["_step"]


    # Pivot data to have strategies as columns
    pivot_table = df_mean.pivot(index='_runtime', columns='strategy_name', values=metric)
    pivot_table['the_index'] = range(len(pivot_table))
    pivot_table_std = df_std.pivot(index='_step', columns='strategy_name', values=metric)

    # akima for gpu.0.gpu and cpu, linear for gpu.0.temp
    pivot_table.interpolate(method='linear', inplace=True, limit=50)
    pivot_table_std.interpolate(method='akima', inplace=True, limit=50)

    if metric == "system.gpu.0.gpu":
        # Number of NaN values to fill
        n_to_fill = 100

        # Iterate through the DataFrame to identify sequences of NaNs
        fill_count = 0
        for index, value in zip(pivot_table['the_index'], pivot_table['Cumulative']):
            if np.isnan(value):
                fill_count += 1
            else:
                if fill_count > 0 and fill_count <= n_to_fill:
                    start_index = index - fill_count
                    pivot_table.iloc[start_index:index, pivot_table.columns.get_loc('Cumulative')] = pivot_table.iloc[start_index-50:index-50, pivot_table.columns.get_loc('Cumulative')]
                fill_count = 0

    # Plotting

    pivot_table.drop(columns="the_index", inplace=True)

    plt.figure(figsize=(12, 6))

    # Define colors for each strategy
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    # Loop through each strategy
    for i, strategy in enumerate(pivot_table.columns):
        means = pivot_table[strategy]
        stds = pivot_table_std[strategy] 
        x = pivot_table.index / (60*60)
        # Plot means for this strategy with customizations
        plt.plot(x, means,
                label=strategy,
                linewidth=2,
                color=colors[i % len(colors)])
        
        # Add variance as shadowed region
        plt.fill_between(x, means - stds, means + stds,
                        color=colors[i % len(colors)], alpha=0.2)

    # Adding labels and title
    plt.xlabel('Runtime (h)')
    plt.ylabel(f'Mean {description}')
    plt.title(f'Mean {description} with Standard Deviation for Different Strategies')
    plt.legend()

    # Show plot
    plt.show()


def extract_energy_consumption(project_name, metric):
    # Login to the wandb
    wandb.login()
    # Extract the metrics from WandB after all the runs
    api = wandb.Api()
    runs = api.runs(project_name)
    # Initialize empty DataFrame
    data = pd.DataFrame()

    # Fetch the logged metrics for each run
    for run in runs:
        history = run.history(stream="system")
        history['run_id'] = run.id
        history['strategy_name'] = run.name
        data = pd.concat([data, history], ignore_index=True)

    data.to_excel("system_metrics.xlsx", index=False)

    # Calculate mean gpu_usage per runtime for each strategy
    data.interpolate(inplace=True)


    # Create a helper column to detect changes in strategy_name
    data['index_run'] = data.groupby("run_id").cumcount()

    df_mean= data.groupby(["strategy_name", "index_run"]).agg(
    mean_metric=(metric, 'mean'),
    std_metric=(metric, 'std'),
    mean_runtime=('_runtime', 'mean')
    ).reset_index()

    # Pivot data to have strategies as columns
    pivot_table = df_mean.pivot(index='mean_runtime', columns='strategy_name', values='mean_metric')
    pivot_table.interpolate(inplace=True, limit=30)

    pivot_table_std = df_mean.pivot(index='mean_runtime', columns='strategy_name', values='std_metric')
    pivot_table_std.interpolate(inplace=True, limit=30)

    # Lists to store area and standard deviation values
    energy_list = []
    std_devs = []

    # Loop through each strategy
    for strategy in pivot_table.columns:
        means = pivot_table[strategy].dropna()
        x = means.index

        stds = pivot_table_std[strategy].dropna()
        x_stds = stds.index
        
        # Calculate area under the curve using the trapezoidal rule
        energy = np.trapz(means, x) / 1e6
        std_dev = np.trapz(stds, x_stds)
        
        # Append area and standard deviation values to the lists
        energy_list.append(energy)
        std_devs.append(std_dev)

    # Create a DataFrame to hold the data for the bar plot
    bar_data = pd.DataFrame({
        'Strategy': pivot_table.columns,
        'Area': energy_list,
        'Std Dev': std_devs
    })

    # Plotting
    plt.figure(figsize=(10, 6))

    # Bar plot for energy_list and standard deviations
    bars = plt.bar(bar_data['Strategy'], bar_data['Area'], capsize=5)

    # Bar plot for energy_list and standard deviations
    plt.bar(bar_data['Strategy'], bar_data['Area'], capsize=5)

    # Adding labels and title
    plt.xlabel('Strategy')
    plt.ylabel('Energy (MJ)')
    plt.title('Energy used for training for different strategies')
    plt.xticks(rotation=45, ha='right')

    # Adding text labels above the bars
    for bar, energy in zip(bars, bar_data['Area']):
        plt.annotate(f'{energy:.2f} MJ', # Text label
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()), # Position
                    xytext=(0, 3),  # Offset from the top of the bar
                    textcoords='offset points',
                    ha='center', va='bottom') # Text alignment

    # Show plot
    plt.tight_layout()
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

    wandb.finish()




def extract_convergence(project_name):
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
    for num_runs, run in enumerate(runs, 1):
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
    strategies = ['Cumulative', 'EWC', 'GEM', 'CWR', 'Naive']

    # Calculate the mean and standard deviation for each strategy
    summary = {}
    fig, ax = plt.subplots(figsize=(12, 6))
    for strategy in strategies:
        # For mean and standard deviation of sum of epochs
        columns = [col for col in df.columns if col.startswith(strategy)]
        strategy_data = df[columns]
        values = strategy_data.values
        mean = values.mean()
        std = values.std()
        summary[strategy] = {'mean': mean, 'std': std}


        # For number of epochs
        strategy_data = []
        for i in range(num_runs):
            Cumulative = f'{strategy}{i}'
            if Cumulative in df:
                strategy_data.extend(df[Cumulative])
        plt.plot(range(len(strategy_data)), strategy_data, marker='o', label=strategy)

    # For number of epochs
    # plt.axhline(y=PATIENCE, color='red', linewidth=2, linestyle='--', label='Patience')
    plt.xlabel('Experience')
    plt.ylabel('Number of epochs')
    plt.title('Number of epochs in each experience for class incremental scenario')
    # Set custom tick labels
    total_iterations = num_runs * 10  # 10 runs, 10 counts for each experience
    tick_positions = list(range(total_iterations))
    tick_labels = [str(i % 10) for i in range(total_iterations)]
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
    plt.title('Mean and Standard Deviation of number of epochs to convergence')
    plt.xlabel('Strategy')
    plt.ylabel('Number of epochs')

    # Adding numerical values at the top of each bar
    for bar, value in zip(bars, summary_df['mean']):
        plt.annotate(f'{value:.2f}', xy=(bar.get_x() + bar.get_width() / 3, value),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

    # Show plot
    plt.savefig("strategies.png")

    # Print the summary DataFrame
    print(summary_df)


    wandb.finish()

    print("finished extracting convergence")


def train_without_ES(group_name, num_runs, scenario):

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

    # Load the CORe50 dataset
    if scenario == "ni":
        core50 = CORe50(scenario="ni", mini=True, object_lvl=True)
    elif scenario == "nc":
        # Load the CORe50 dataset
        core50_train = CORe50Dataset(train=True, mini=True)
        core50_test = CORe50Dataset(train=False, mini=True)
        # Create different split
        core50 = nc_benchmark(core50_train, core50_test, n_experiences=10, shuffle=False, task_labels=False, train_transform=transform(), eval_transform=transform())
    else:
        raise NameError("Scenario name unknown")

    # the task label of each train_experience.
    print('--- Task labels:')
    print(core50.task_labels)

    # Instatiate train, validation and test streams

    train_stream = core50.train_stream
    test_stream = core50.test_stream

    criterion = CrossEntropyLoss()
    lr = 0.001
    epochs = EPOCHS
    batchsize_train = 128
    batchsize_eval = 128


    generator_strategy = Generator_Strategy_without_ES(N_CLASSES, lr, batchsize_train, batchsize_eval,\
                    epochs, device, 2)

    for j in range(num_runs):

        loggers = []
        loggers.append(WandBLogger(project_name=project_name, run_name="Naive", params={"reinit": True, "group": group_name}))
        loggers.append(InteractiveLogger())
        loggers.append(TextLogger(open(f'Naive_{j}.txt', 'a')))

        eval_plugin = EvaluationPlugin(
        loss_metrics(epoch=True, stream=True),
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

        model = SimpleCNN(num_classes=50).to(device)
        print(f"Main model {model}")
        optimizer = Adam(model.parameters(), lr=lr)

        cl_strategy = Naive(
            model, optimizer, criterion, device=device,
            train_mb_size=batchsize_train, train_epochs=epochs, eval_mb_size=batchsize_eval, evaluator=eval_plugin)
        

        print(f"Current training strategy: {cl_strategy}")
        for train_experience in train_stream:
            print(f"Experience number train {train_experience.current_experience}")
            print(f"classes in this experience train {train_experience.classes_in_this_experience}")
            print(f"Classes seen so far train {train_experience.classes_seen_so_far}")
            print(f"Training on train {len(train_experience.dataset)} examples")

            cl_strategy.train(train_experience)

            cl_strategy.eval(test_stream)


        loggers = []
        loggers.append(WandBLogger(project_name=project_name, run_name="CWR*", params={"reinit": True, "group": group_name}))
        loggers.append(InteractiveLogger())
        loggers.append(TextLogger(open(f'CWR*_{j}.txt', 'a')))

        eval_plugin = EvaluationPlugin(
        loss_metrics(epoch=True, stream=True),
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


        model = SimpleCNN(num_classes=50).to(device)
        print(f"Main model {model}")
        optimizer = Adam(model.parameters(), lr=lr)
        
        cl_strategy = CWRStar(
            model, optimizer, criterion, cwr_layer_name='classifier.0', device=device,
            train_mb_size=batchsize_train, train_epochs=epochs, eval_mb_size=batchsize_eval, evaluator=eval_plugin)

        print(f"Current training strategy: {cl_strategy}")
        for train_experience in train_stream:
            print(f"Experience number train {train_experience.current_experience}")
            print(f"classes in this experience train {train_experience.classes_in_this_experience}")
            print(f"Classes seen so far train {train_experience.classes_seen_so_far}")
            print(f"Training on train {len(train_experience.dataset)} examples")

            cl_strategy.train(train_experience)

            cl_strategy.eval(test_stream)



        loggers = []
        loggers.append(WandBLogger(project_name=project_name, run_name="GEM", params={"reinit": True, "group": group_name}))
        loggers.append(InteractiveLogger())
        loggers.append(TextLogger(open(f'GEM_{j}.txt', 'a')))

        eval_plugin = EvaluationPlugin(
        loss_metrics(epoch=True, stream=True),
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


        model = SimpleCNN(num_classes=50).to(device)
        print(f"Main model {model}")
        optimizer = Adam(model.parameters(), lr=lr)

        cl_strategy = GEM(
            model, optimizer, criterion, device=device, patterns_per_exp=1024, memory_strength=1,
            train_mb_size=batchsize_train, train_epochs=epochs, eval_mb_size=batchsize_eval, evaluator=eval_plugin)

        print(f"Current training strategy: {cl_strategy}")
        for train_experience in train_stream:
            print(f"Experience number train {train_experience.current_experience}")
            print(f"classes in this experience train {train_experience.classes_in_this_experience}")
            print(f"Classes seen so far train {train_experience.classes_seen_so_far}")
            print(f"Training on train {len(train_experience.dataset)} examples")

            cl_strategy.train(train_experience)

            cl_strategy.eval(test_stream)

        
        loggers = []
        loggers.append(WandBLogger(project_name=project_name, run_name="EWC", params={"reinit": True, "group": group_name}))
        loggers.append(InteractiveLogger())
        loggers.append(TextLogger(open(f'EWC_{j}.txt', 'a')))

        eval_plugin = EvaluationPlugin(
        loss_metrics(epoch=True, stream=True),
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

        model = SimpleCNN(num_classes=50).to(device)
        print(f"Main model {model}")
        optimizer = Adam(model.parameters(), lr=lr)

        cl_strategy = EWC(
            model, optimizer, criterion, ewc_lambda=100, device=device,
            train_mb_size=batchsize_train, train_epochs=epochs, eval_mb_size=batchsize_eval, evaluator=eval_plugin)

        print(f"Current training strategy: {cl_strategy}")
        for train_experience in train_stream:
            print(f"Experience number train {train_experience.current_experience}")
            print(f"classes in this experience train {train_experience.classes_in_this_experience}")
            print(f"Classes seen so far train {train_experience.classes_seen_so_far}")
            print(f"Training on train {len(train_experience.dataset)} examples")

            cl_strategy.train(train_experience)

            cl_strategy.eval(test_stream)



        loggers = []
        loggers.append(WandBLogger(project_name=project_name, run_name="GenerativeReplay", params={"reinit": True, "group": group_name}))
        loggers.append(InteractiveLogger())
        loggers.append(TextLogger(open(f'GenerativeReplay_{j}.txt', 'a')))

        eval_plugin = EvaluationPlugin(
        loss_metrics(epoch=True, stream=True),
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


        model = SimpleCNN(num_classes=50).to(device)
        print(f"Main model {model}")
        optimizer = Adam(model.parameters(), lr=lr)

        cl_strategy = GenerativeReplay(
            model, optimizer, criterion, device=device,
            train_mb_size=batchsize_train, train_epochs=epochs, eval_mb_size=batchsize_eval, evaluator=eval_plugin, generator_strategy=generator_strategy)

        print(f"Current training strategy: {cl_strategy}")
        for train_experience in train_stream:
            print(f"Experience number train {train_experience.current_experience}")
            print(f"classes in this experience train {train_experience.classes_in_this_experience}")
            print(f"Classes seen so far train {train_experience.classes_seen_so_far}")
            print(f"Training on train {len(train_experience.dataset)} examples")

            cl_strategy.train(train_experience)

            cl_strategy.eval(test_stream)



        loggers = []
        loggers.append(WandBLogger(project_name=project_name, run_name="Cumulative", params={"reinit": True, "group": group_name}))
        loggers.append(InteractiveLogger())
        loggers.append(TextLogger(open(f'Cumulative_{j}.txt', 'a')))

        eval_plugin = EvaluationPlugin(
        loss_metrics(epoch=True, stream=True),
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


        model = SimpleCNN(num_classes=50).to(device)
        print(f"Main model {model}")
        optimizer = Adam(model.parameters(), lr=lr)

        cl_strategy = Cumulative(
            model, optimizer, criterion, device=device,
            train_mb_size=batchsize_train, train_epochs=epochs, eval_mb_size=batchsize_eval, evaluator=eval_plugin)

        print(f"Current training strategy: {cl_strategy}")
        for train_experience in train_stream:
            print(f"Experience number train {train_experience.current_experience}")
            print(f"classes in this experience train {train_experience.classes_in_this_experience}")
            print(f"Classes seen so far train {train_experience.classes_seen_so_far}")
            print(f"Training on train {len(train_experience.dataset)} examples")

            cl_strategy.train(train_experience)

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
    batchsize_train = 512
    batchsize_eval = 512
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
    batchsize_train = 512
    batchsize_eval = 512

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
    batchsize_train = 512
    batchsize_eval = 512

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
    batchsize_train = 512
    batchsize_eval = 512

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
    core50 = CORe50(scenario="ni", mini=True, object_lvl=False)

    # the task label of each train_experience.
    print('--- Task labels:')
    print(core50.task_labels)

    # Instatiate train, validation and test streams
    core50 = benchmark_with_validation_stream(core50, 0.2)
    train_stream = core50.train_stream
    val_stream = core50.valid_stream


    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = CrossEntropyLoss()

    loggers = []
    loggers.append(InteractiveLogger())

    eval_plugin = EvaluationPlugin(
    loss_metrics(epoch=True, stream=True),
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





    # model:
    generator = MlpVAE((3, 32, 32), nhid=2, device=device)
    # optimzer:
    lr = 0.001

    optimizer_generator = Adam(
        filter(lambda p: p.requires_grad, generator.parameters()),
        lr=lr,
        weight_decay=0.0001,
    )
    # strategy (with plugin):
    generator_strategy = VAETraining(
        model=generator,
        optimizer=optimizer_generator,
        train_mb_size=100,
        train_epochs=4,
        eval_mb_size=100,
        device=device,
        plugins=[
            GenerativeReplayPlugin(
                replay_size=None,
                increasing_replay_size=False,
            )
        ],
    )


    # CREATE THE STRATEGY INSTANCE (GenerativeReplay)
    cl_strategy = GenerativeReplay(
        model,
        optimizer,
        criterion,
        train_mb_size=20,
        train_epochs=4,
        eval_mb_size=20,
        device=device,
        evaluator=eval_plugin,
        eval_every=1, 
        generator_strategy=generator_strategy)

    # TRAINING LOOP
    print("Starting experiment...")

    for train_experience in train_stream:
        print("Start of train_experience ", train_experience.current_experience)
        print(f"classes in this experience train {train_experience.classes_in_this_experience}")
        cl_strategy.train(train_experience, eval_streams=[val_stream])
        print("Training completed")





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


def progressive_neural_network(group_name, ES_metric, scenario):

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

    # Load the CORe50 dataset
    if scenario == "ni":
        core50 = CORe50(scenario="ni", mini=True, object_lvl=True)
    elif scenario == "nc":
        # Load the CORe50 dataset
        core50_train = CORe50Dataset(train=True, mini=True)
        core50_test = CORe50Dataset(train=False, mini=True)
        # Create different split
        core50 = nc_benchmark(core50_train, core50_test, n_experiences=10, shuffle=False, task_labels=False, train_transform=transform(), eval_transform=transform())
    else:
        raise NameError("Scenario name unknown")

    # Instatiate train, validation and test streams
    validation_size = 0.2
    foo = lambda exp: class_balanced_split_strategy(validation_size, exp)
    core50 = benchmark_with_validation_stream(core50, custom_split_strategy=foo)

    train_stream = core50.train_stream
    val_stream = core50.valid_stream
    test_stream = core50.test_stream

    criterion = CrossEntropyLoss()
    lr = 0.001
    epochs = 50
    batchsize_train = 128
    batchsize_eval = 128
    patience = PATIENCE
    eval_every = 2
    if ES_metric == "Top1_Acc_Stream":
        min_max = "max"
    elif ES_metric == "Loss_Stream":
        min_max = "min"
    else:
        raise NameError

    loggers = []
    loggers.append(WandBLogger(project_name=project_name, run_name="Test_PNN", params={"reinit": True, "group": group_name}))
    loggers.append(InteractiveLogger())
    loggers.append(TextLogger(open(f'Test_PNN.txt', 'a')))

    eval_plugin = EvaluationPlugin(
    loss_metrics(epoch=True, stream=True),
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

    # model = SimpleCNN(num_classes=50).to(device)
    model = PNN(num_layers=1, in_features=32*32)
    print(f"Main model {model}")
    optimizer = Adam(model.parameters(), lr=lr)

    cl_strategy = PNNStrategy(
        model, optimizer, criterion,
        train_mb_size=batchsize_train, train_epochs=epochs, eval_mb_size=batchsize_eval, evaluator=eval_plugin, \
        eval_every=eval_every, plugins=[EarlyStoppingPlugin(patience, "valid", ES_metric, min_max)])
    

    print(f"Current training strategy: {cl_strategy}")
    for i, train_experience in enumerate(train_stream, 1):
        # Accumulate the experiences
        val_experience = val_stream[:i]
        print(f"Experience number train {train_experience.current_experience}")
        print(f"classes in this experience train {train_experience.classes_in_this_experience}")
        print(f"Classes seen so far train {train_experience.classes_seen_so_far}")
        print(f"Training on train {len(train_experience.dataset)} examples")

        cl_strategy.train(train_experience, eval_streams=[val_experience])

    cl_strategy.eval(test_stream)



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
    project_name = "New_Classes_50_classes_ES_ACC_patience_25_batchsize128_lr0.001_44_epochs_resource_efficiency"
    # project_name = "New_Classes_50_classes_ES_ACC_patience_25_batchsize128_lr0.001_50_epochs_eval_like_train"
    # project_name = "New_Instances_10_classes"
    # ES_metric = ["Top1_Acc_Stream", "Loss_Stream"]

    # train_without_ES(project_name, 5, "nc")
    # train_with_ES_evaluation(project_name, 1, ES_metric[0], "nc")
    # progressive_neural_network(project_name, ES_metric[0], "nc")
    # core50_check()
    # GR_GR()
    # train_iteratively_ES(project_name, 1)
    # cumulative_only(project_name, 5)
    # cumulative_via_plugin(project_name, 5)
    # cumulative_vs_naive(project_name, 1)
    # concat_cumulative_naive(project_name, 1)
    # extract_metrics(project_name)
    # # extract_accuracy_valid(project_name)
    # extract_convergence(project_name)
    # extract_system_metrics(project_name, "system.gpu.0.gpu", "GPU Utilization")
    # extract_system_metrics(project_name, "system.gpu.0.temp", "GPU Temperature (°C)")
    # extract_system_metrics(project_name, "system.cpu", "CPU Utilization (%)")
    # extract_system_metrics(project_name, "system.memory", "System Memory Utilization (%)")
    extract_metrics(project_name, "Top1_Acc_Stream/eval_phase/test_stream/Task000", "Accuracy (%) on Train Set")
    # extract_GPU_metrics(project_name, "system.gpu.0.powerWatts", "GPU Power Usage")
    # extract_energy_consumption(project_name, "system.gpu.0.powerWatts")
    # project_name = "CWRStar_hiperparameter_search_50_classes_ni_bs128"
    # CWRStar_hiperparameter_search(project_name, ES_metric[0], "max", "ni")
    # project_name = "GEM_hiperparameter_search_50_classes_ni_bs128"
    # GEM_hiperparemeter_search(project_name, ES_metric[0], "max", "ni")
    # project_name = "EWC_hiperparameter_search_50_classes_ni_bs128"
    # EWC_hiperparameter_search(project_name, ES_metric[0], "max", "ni")
    # project_name = "GR_hiperparameter_search_50_classes_ni_bs128"
    # GR_hiperparameter_search(project_name, ES_metric[0], "max", "ni")
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
