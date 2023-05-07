from avalanche.benchmarks.classic import CORe50
from avalanche.benchmarks.utils import AvalancheDataset
import torch
from avalanche.evaluation.metrics import forgetting_metrics, \
accuracy_metrics, loss_metrics, timing_metrics, cpu_usage_metrics, \
confusion_matrix_metrics, disk_usage_metrics, gpu_usage_metrics, ram_usage_metrics
from avalanche.training.plugins import EvaluationPlugin
from avalanche.logging import InteractiveLogger, TextLogger, TensorboardLogger, WandBLogger
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from avalanche.models import SimpleCNN, MlpVAE, MTSimpleCNN, SimpleMLP, SlimResNet18, PNN, MLPAdapter
from avalanche.training.supervised import Naive, Cumulative, EWC, GenerativeReplay, VAETraining, Replay, GEM, PNNStrategy, CWRStar
from avalanche.training.plugins import GenerativeReplayPlugin
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import pandas as pd 
import wandb


def main(n_strategies):
  
    # Load the CORe50 dataset
    core50 = CORe50(scenario="nc", mini=True, object_lvl=True)

    # the task label of each experience.
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
    # model = SimpleCNN(num_classes=50).to(device)
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
    epochs = 50
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
        for experience in train_stream:
            print(f"Experience number {experience.current_experience}")
            print(f"Classes seen so far {experience.classes_seen_so_far}")
            print(f"Training on {len(experience.dataset)} examples")
   
            cl_strategy.train(experience)

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
    for experience in benchmark.train_stream:
        print("Start of experience ", experience.current_experience)
        cl_strategy.train(experience)
        print("Training completed")

        samples = model.generate(10)
        samples = samples.detach().cpu().numpy()

        for j in range(10):
            axarr[k, j].imshow(samples[j, 0], cmap="gray")
            axarr[k, 4].set_title("Generated images for experience " + str(k))
        np.vectorize(lambda ax: ax.axis("off"))(axarr)
        k += 1

    f.subplots_adjust(hspace=1.2)
    plt.savefig("VAE_output_per_exp")
    plt.show()



def main2():

    # Load the CORe50 dataset
    core50 = CORe50(scenario="nc", mini=True, object_lvl=True)

    # the task label of each experience.
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
    epochs = 100
    batchsize_train = 100
    batchsize_eval = 100

    logger = WandBLogger(project_name="avalanche", run_name="Naive", params={"reinit": True, "group": "Experiment_1"})

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(epoch=True),
        cpu_usage_metrics(epoch=True),
        gpu_usage_metrics(gpu_id=0, epoch=True),
        ram_usage_metrics(epoch=True),
        disk_usage_metrics(epoch=True),
        loggers=logger,
        strict_checks=False
    )
    
    cl_strategy = Naive(
        model, optimizer, criterion, device=device,
        train_mb_size=batchsize_train, train_epochs=epochs, eval_mb_size=batchsize_eval, evaluator=eval_plugin)
       
    print(f"Current training strategy: {cl_strategy}")

    for experience in train_stream:
        print(f"Experience number {experience.current_experience}")
        print(f"Classes seen so far {experience.classes_seen_so_far}")
        print(f"Training on {len(experience.dataset)} examples")

        cl_strategy.train(experience)

    # Evaluate on test set
    print(f"Testing on {len(test_stream[0].dataset)} examples")
    results.append(cl_strategy.eval(test_stream))
    print(f"Evaluation on test stream:\n {results}")


    logger = WandBLogger(project_name="avalanche", run_name="CWRStar", params={"reinit": True, "group": "Experiment_1"})

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(epoch=True),
        cpu_usage_metrics(epoch=True),
        gpu_usage_metrics(gpu_id=0, epoch=True),
        ram_usage_metrics(epoch=True),
        disk_usage_metrics(epoch=True),
        loggers=logger,
        strict_checks=False
    )
    
    cl_strategy = CWRStar(
        model, optimizer, criterion, cwr_layer_name=None, device=device,
        train_mb_size=batchsize_train, train_epochs=epochs, eval_mb_size=batchsize_eval, evaluator=eval_plugin
     )

    print(f"Current training strategy: {cl_strategy}")
    results = []
    for experience in train_stream:
        print(f"Experience number {experience.current_experience}")
        print(f"Classes seen so far {experience.classes_seen_so_far}")
        print(f"Training on {len(experience.dataset)} examples")

        cl_strategy.train(experience)

    # Evaluate on test set
    print(f"Testing on {len(test_stream[0].dataset)} examples")
    results.append(cl_strategy.eval(test_stream))
    print(f"Evaluation on test stream:\n {results}")


    logger = WandBLogger(project_name="avalanche", run_name="GEM", params={"reinit": True, "group": "Experiment_1"})

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(epoch=True),
        cpu_usage_metrics(epoch=True),
        gpu_usage_metrics(gpu_id=0, epoch=True),
        ram_usage_metrics(epoch=True),
        disk_usage_metrics(epoch=True),
        loggers=logger,
        strict_checks=False
    )
    
    cl_strategy = GEM(
        model, optimizer, criterion, device=device,
        train_mb_size=batchsize_train, train_epochs=epochs, eval_mb_size=batchsize_eval, evaluator=eval_plugin,
        patterns_per_exp=8
     )

    print(f"Current training strategy: {cl_strategy}")
    results = []
    for experience in train_stream:
        print(f"Experience number {experience.current_experience}")
        print(f"Classes seen so far {experience.classes_seen_so_far}")
        print(f"Training on {len(experience.dataset)} examples")

        cl_strategy.train(experience)

    # Evaluate on test set
    print(f"Testing on {len(test_stream[0].dataset)} examples")
    results.append(cl_strategy.eval(test_stream))
    print(f"Evaluation on test stream:\n {results}")
    

    logger = WandBLogger(project_name="avalanche", run_name="EWC", params={"reinit": True, "group": "Experiment_1"})

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(epoch=True),
        cpu_usage_metrics(epoch=True),
        gpu_usage_metrics(gpu_id=0, epoch=True),
        ram_usage_metrics(epoch=True),
        disk_usage_metrics(epoch=True),
        loggers=logger,
        strict_checks=False
    )

    cl_strategy = EWC(
        model, optimizer, criterion, ewc_lambda=1.0, device=device,
        train_mb_size=batchsize_train, train_epochs=epochs, eval_mb_size=batchsize_eval, evaluator=eval_plugin
    )

    print(f"Current training strategy: {cl_strategy}")
    results = []
    for experience in train_stream:
        print(f"Experience number {experience.current_experience}")
        print(f"Classes seen so far {experience.classes_seen_so_far}")
        print(f"Training on {len(experience.dataset)} examples")

        cl_strategy.train(experience)

    # Evaluate on test set
    print(f"Testing on {len(test_stream[0].dataset)} examples")
    results.append(cl_strategy.eval(test_stream))
    print(f"Evaluation on test stream:\n {results}")

    logger = WandBLogger(project_name="avalanche", run_name="Cumulative", params={"reinit": True, "group": "Experiment_1"})

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(epoch=True),
        cpu_usage_metrics(epoch=True),
        gpu_usage_metrics(gpu_id=0, epoch=True),
        ram_usage_metrics(epoch=True),
        disk_usage_metrics(epoch=True),
        loggers=logger,
        strict_checks=False
    )

    cl_strategy = Cumulative(
        model, optimizer, criterion, device=device,
        train_mb_size=batchsize_train, train_epochs=epochs, eval_mb_size=batchsize_eval, evaluator=eval_plugin
     )

    print(f"Current training strategy: {cl_strategy}")
    results = []
    for experience in train_stream:
        print(f"Experience number {experience.current_experience}")
        print(f"Classes seen so far {experience.classes_seen_so_far}")
        print(f"Training on {len(experience.dataset)} examples")

        cl_strategy.train(experience)

    # Evaluate on test set
    print(f"Testing on {len(test_stream[0].dataset)} examples")
    results.append(cl_strategy.eval(test_stream))
    print(f"Evaluation on test stream:\n {results}")


def gem(it=5, start_ppe=4):

    # Load the CORe50 dataset
    core50 = CORe50(scenario="nc", mini=True, object_lvl=True)

    # the task label of each experience.
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

        print(f"Patterns per experience: {patterns_per_experience}")
        cl_strategy = GEM(
            model, optimizer, criterion, device=device,
            train_mb_size=batchsize_train, train_epochs=epochs, eval_mb_size=batchsize_eval, evaluator=eval_plugin,
            patterns_per_exp=patterns_per_experience
        )

        print(f"Current training strategy: {cl_strategy}")
        results = []
        for experience in train_stream:
            print(f"Experience number {experience.current_experience}")
            print(f"Classes seen so far {experience.classes_seen_so_far}")
            print(f"Training on {len(experience.dataset)} examples")

            cl_strategy.train(experience)

        # Evaluate on test set
        print(f"Testing on {len(test_stream[0].dataset)} examples")

        print(f"Evaluation on test stream {cl_strategy.eval(test_stream)}")



def ewc(it=5, ewc_lambda=0.25):

    # Load the CORe50 dataset
    core50 = CORe50(scenario="nc", mini=True, object_lvl=True)

    # the task label of each experience.
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
        for experience in train_stream:
            print(f"Experience number {experience.current_experience}")
            print(f"Classes seen so far {experience.classes_seen_so_far}")
            print(f"Training on {len(experience.dataset)} examples")

            cl_strategy.train(experience)

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
            minibatch=True, epoch=True, experience=True, stream=True
        ),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True),
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
    for experience in benchmark.train_stream:
        print("Start of experience ", experience.current_experience)
        cl_strategy.train(experience)
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
        minibatch=True, epoch=True, experience=True, stream=True
    ),
    loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    forgetting_metrics(experience=True),
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
    for experience in benchmark.train_stream:
        print("Start of experience ", experience.current_experience)
        cl_strategy.train(experience)
        print("Training completed")

        samples = model.generate(10)
        samples = samples.detach().cpu().numpy()

        for j in range(10):
            axarr[k, j].imshow(samples[j, 0], cmap="gray")
            axarr[k, 4].set_title("Generated images for experience " + str(k))
        np.vectorize(lambda ax: ax.axis("off"))(axarr)
        k += 1

    f.subplots_adjust(hspace=1.2)
    plt.savefig("VAE_output_per_exp")
    plt.show()



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
    main2()
    # wandb_import()
    # GR_Plugin()
    # GR_GR()
    # GR_Plugin_sandbox()
    # test_tbplugin()
    # gem()
    # ewc()

# [VAETraining(
#     model, optimizer, device=device,
#     train_mb_size=batchsize_train, train_epochs=epochs, eval_mb_size=batchsize_eval, evaluator=eval_plugin, 
#     plugins=[GenerativeReplayPlugin()]
# ),
