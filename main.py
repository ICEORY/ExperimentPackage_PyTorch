from opt import *
import models as MD
import time
from dataloader import *
from trainer import *
from visualization import *
from termcolor import colored
import torch
import torch.backends.cudnn as cudnn
from checkpoint import *
import random
import datetime
from utils import *


def main(net_opt=None):
    """requirements:
    apt-get install graphviz
    pip install pydot termcolor"""

    start_time = time.time()
    opt = net_opt or NetOption()

    # set torch seed
    # init random seed
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed(opt.manualSeed)
    cudnn.benchmark = True
    if opt.nGPU == 1 and torch.cuda.device_count() >= 1:
        assert opt.GPU <= torch.cuda.device_count()-1, "Invalid GPU ID"
        torch.cuda.set_device(opt.GPU)
    else:
        torch.cuda.set_device(opt.GPU)

    # create data loader
    data_loader = DataLoader(dataset=opt.data_set, train_batch_size=opt.trainBatchSize,
                             test_batch_size=opt.testBatchSize,
                             n_threads=opt.nThreads, ten_crop=opt.tenCrop)
    train_loader, test_loader = data_loader.getloader()

    # define check point
    check_point = CheckPoint(opt=opt)
    # create residual network model
    if opt.retrain:
        check_point_params = check_point.retrainmodel()
    elif opt.resume:
        check_point_params = check_point.resumemodel()
    else:
        check_point_params = check_point.check_point_params

    optimizer = check_point_params['opts']
    start_epoch = check_point_params['resume_epoch'] or 0
    if check_point_params['resume_epoch'] is not None:
        start_epoch += 1
    if start_epoch >= opt.nEpochs:
        start_epoch = 0
    if opt.netType == "ResNet":
        model = check_point_params['model'] or MD.ResNet(depth=opt.depth, num_classes=opt.nClasses,
                                                         wide_factor=opt.wideFactor)
        model = dataparallel(model, opt.nGPU, opt.GPU)
    elif opt.netType == "PreResNet":
        model = check_point_params['model'] or MD.PreResNet(depth=opt.depth, num_classes=opt.nClasses,
                                                            wide_factor=opt.wideFactor)
        model = dataparallel(model, opt.nGPU, opt.GPU)
    elif opt.netType == "LeNet5":
        model = check_point_params['model'] or MD.LeNet5()
        model = dataparallel(model, opt.nGPU, opt.GPU)

    else:
        assert False, "invalid net type"

    # create online board
    if opt.onlineBoard:
        try:
            online_board = BoardManager("main")
        except:
            online_board = None
            print "|===> Failed to create online board! Check whether you have ran <python -m visdom.server>"
    else:
        online_board = None

    trainer = Trainer(model=model, opt=opt, optimizer=optimizer, online_board=online_board)
    print "|===>Create trainer"

    # define visualizer
    visualize = Visualization(opt=opt)
    visualize.writeopt(opt=opt)
    # visualize model
    if opt.drawNetwork:
        if opt.data_set == "cifar10" or opt.data_set == "cifar100":
            rand_input = torch.randn(1, 3, 32, 32)
        elif opt.data_set == "mnist":
            rand_input = torch.randn(1, 1, 28, 28)
        else:
            assert False, "invalid data set"
        rand_input = Variable(rand_input.cuda())
        rand_output = trainer.forward(rand_input)
        visualize.gennetwork(rand_output)
        visualize.savenetwork()

    # test model
    if opt.testOnly:
        trainer.test(epoch=0, test_loader=test_loader)
        return

    best_top1 = 100
    best_top5 = 100
    for epoch in range(start_epoch, opt.nEpochs):
        start_epoch = 0
        # training and testing
        train_error, train_loss, train5_error = trainer.train(epoch=epoch, train_loader=train_loader)
        test_error, test_loss, test5_error = trainer.test(epoch=epoch, test_loader=test_loader)

        # show training information on online board
        if online_board is not None:
            online_board.updateplot(train_error, train5_error, train_loss, mode="Train")
            online_board.updateplot(test_error, test5_error, test_loss, mode="Test")

        # write and print result
        log_str = "%d\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t" % (epoch, train_error, train_loss, test_error,
                                                                test_loss, train5_error, test5_error)
        visualize.writelog(log_str)
        best_flag = False
        if best_top1 >= test_error:
            best_top1 = test_error
            best_top5 = test5_error
            best_flag = True
            if online_board is not None:
                online_board.updateresult([best_top1, best_top5, test_loss])
            print colored("==>Best Result is: Top1 Error: %f, Top5 Error: %f\n" % (best_top1, best_top5)
                          , "red")
        else:
            print colored("==>Best Result is: Top1 Error: %f, Top5 Error: %f\n" % (best_top1, best_top5)
                          , "blue")

        # save check_point
        # save best result and recent state
        check_point.savemodel(epoch=epoch, model=trainer.model,
                              opts=trainer.optimzer, best_flag=best_flag)

        if (epoch+1) % opt.drawInterval == 0:
            visualize.drawcurves()

    end_time = time.time()
    time_interval = end_time-start_time

    t_string = "Running Time is: "+str(datetime.timedelta(seconds=time_interval)) + "\n"
    print(t_string)

    # save experimental results
    visualize.writereadme("Best Result of all is: Top1 Error: %f, Top5 Error: %f\n" % (best_top1, best_top5))
    visualize.writereadme(t_string)
    visualize.drawcurves()

if __name__ == '__main__':
    main()
