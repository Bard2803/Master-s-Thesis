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
from avalanche.models import SimpleCNN
from avalanche.training.supervised import Naive, Cumulative, EWC, GenerativeReplay


import GPUtil

gpus = GPUtil.getGPUs()
n_gpus = len(GPUtil.getGPUs())
print(gpus)
print(n_gpus)


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
        loggers=[InteractiveLogger()],
        strict_checks=False
    )

    model = SimpleCNN(num_classes=50).to(device)
    print(model)
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = CrossEntropyLoss()
    epochs = 1
    batchsize_train = 100
    batchsize_eval = 100
    # Cumulative(
    #     model, optimizer, criterion, device=device,
    #     train_mb_size=batchsize_train, train_epochs=epochs, eval_mb_size=batchsize_eval, evaluator=eval_plugin
    # ), 
    cl_strategies = [
    EWC(
        model, optimizer, criterion, ewc_lambda=1.0, device=device,
        train_mb_size=batchsize_train, train_epochs=epochs, eval_mb_size=batchsize_eval, evaluator=eval_plugin
    ), 
    GenerativeReplay(
        model, optimizer, criterion, device=device,
        train_mb_size=batchsize_train, train_epochs=epochs, eval_mb_size=batchsize_eval, evaluator=eval_plugin
    ),
    Naive(
        model, optimizer, criterion, device=device,
        train_mb_size=batchsize_train, train_epochs=epochs, eval_mb_size=batchsize_eval, evaluator=eval_plugin
    )]

    # TRAINING LOOP
    results_train = []
    results_test = []
    for cl_strategy in cl_strategies:
        print(f"Current training strategy: {cl_strategy}")
        for experience in train_stream:
            print(f"Experience number {experience.current_experience}")
            print(f"Classes seen so far {experience.classes_seen_so_far}")
   
            # Seems there is only one test experience 
            current_test_set = test_stream[0].dataset
            print('This task contains', len(current_test_set), 'test examples')

            cl_strategy.train(experience)

            # results_train.append(cl_strategy.eval(core50.train_stream))
            # results_test.append(cl_strategy.eval(core50.test_stream))
        # print(f'Results for train stream for strategy {cl_strategy}')
        # print(results_train)
        # print(f'Results for test stream for strategy {cl_strategy}')
        # print(results_test)



if __name__ == "__main__":
  main()