from avalanche.benchmarks.classic import CORe50
from avalanche.benchmarks.utils import AvalancheDataset
import torch
from avalanche.evaluation.metrics import forgetting_metrics, \
accuracy_metrics, loss_metrics, timing_metrics, cpu_usage_metrics, \
confusion_matrix_metrics, disk_usage_metrics, gpu_usage_metrics, ram_usage_metrics
from avalanche.training.plugins import EvaluationPlugin
from avalanche.logging import InteractiveLogger, TextLogger, TensorboardLogger
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from avalanche.models import SimpleCNN, MlpVAE, MTSimpleCNN, SimpleMLP, SlimResNet18, PNN
from avalanche.training.supervised import Naive, Cumulative, EWC, GenerativeReplay, VAETraining, Replay, GEM, PNNStrategy
from avalanche.training.plugins import GenerativeReplayPlugin
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn


import GPUtil

gpus = GPUtil.getGPUs()
n_gpus = len(GPUtil.getGPUs())
print(f'The MPS GPUs: {gpus}')
print(f'The MPS number of GPUs {n_gpus}')


def main():
  
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

    device = 'cpu'

    loggers = []
    # log to Tensorboard
    loggers.append(TensorboardLogger())
    # log to text file
    loggers.append(TextLogger(open('log.txt', 'a')))
    # print to stdout
    loggers.append(InteractiveLogger())

    # DEFINE THE EVALUATION PLUGIN
    # The evaluation plugin manages the metrics computation.
    # It takes as argument a list of metrics, collectes their results and returns
    # them to the strategy it is attached to.
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        cpu_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        gpu_usage_metrics(minibatch=True, epoch=True, gpu_id=0, experience=True, stream=True),
        ram_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loggers=loggers,
        strict_checks=False
    )

    model = SimpleCNN(num_classes=50).to(device)

    # model = MlpVAE((3, 32, 32), nhid=2, device=device)

    # model = MTSimpleCNN().to(device)

    # model = SimpleMLP(num_classes=50, input_size=(3,32,32), hidden_size=32, hidden_layers = 2, drop_rate=0.5)

    # model = SlimResNet18(nclasses=50)

    # Define the custom PNN model
    class CustomPNN():
        def __init__(self, base_model):
            super().__init__(base_model)
            self.hidden_size = 100  # Size of the new hidden layer
            
        def progressive_block(self):
            # Add a new hidden layer and classification layer
            new_layers = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.output_size)
            )
            self.add_block(new_layers)



    print(model)
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = CrossEntropyLoss()
    epochs = 1
    batchsize_train = 100
    batchsize_eval = 100

    pnn_model = PNN(num_layers=1, in_features=32*32*3, hidden_features_per_column=32)

    cl_strategies = [PNNStrategy(
        pnn_model, optimizer, criterion, device=device,
        train_mb_size=batchsize_train, train_epochs=epochs, eval_mb_size=batchsize_eval, evaluator=eval_plugin
     ), 
    GEM(
        model, optimizer, criterion, device=device,
        train_mb_size=batchsize_train, train_epochs=epochs, eval_mb_size=batchsize_eval, evaluator=eval_plugin,
        patterns_per_exp=5
     ), 
    Cumulative(
        model, optimizer, criterion, device=device,
        train_mb_size=batchsize_train, train_epochs=epochs, eval_mb_size=batchsize_eval, evaluator=eval_plugin
     ), 
    EWC(
        model, optimizer, criterion, ewc_lambda=1.0, device=device,
        train_mb_size=batchsize_train, train_epochs=epochs, eval_mb_size=batchsize_eval, evaluator=eval_plugin
    ), 
    Naive(
        model, optimizer, criterion, device=device,
        train_mb_size=batchsize_train, train_epochs=epochs, eval_mb_size=batchsize_eval, evaluator=eval_plugin
    )]

    # TRAINING LOOP
    for cl_strategy in cl_strategies:
        print(f"Current training strategy: {cl_strategy}")
        for experience in train_stream:
            print(f"Experience number {experience.current_experience}")
            print(f"Classes seen so far {experience.classes_seen_so_far}")
   
            # Seems there is only one test experience 
            current_test_set = test_stream[0].dataset
            print('This task contains', len(current_test_set), 'test examples')

            cl_strategy.train(experience)

def GR():

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


def GR_GR(args):
    # --- CONFIG
    device = torch.device(
        f"cuda:{args.cuda}"
        if torch.cuda.is_available() and args.cuda >= 0
        else "cpu"
    )

    # --- BENCHMARK CREATION
    benchmark = SplitMNIST(n_experiences=10, seed=1234)
    # ---------

    # MODEL CREATION
    model = SimpleMLP(num_classes=benchmark.n_classes)

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






if __name__ == "__main__":
    main()
    # GR()

# [VAETraining(
#     model, optimizer, device=device,
#     train_mb_size=batchsize_train, train_epochs=epochs, eval_mb_size=batchsize_eval, evaluator=eval_plugin, 
#     plugins=[GenerativeReplayPlugin()]
# ),
