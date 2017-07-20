import torch.nn as nn
import math
import numpy as np
import torch
import datetime
import models as MD
from termcolor import colored


def dataparallel(model, ngpus, gpu0=0):
    if ngpus == 0:
        assert False, "only support gpu mode"
    gpu_list = list(range(gpu0, gpu0+ngpus))
    assert torch.cuda.device_count() >= gpu0+ngpus, "Invalid Number of GPUs"
    if isinstance(model, list):
        for i in range(len(model)):
            if ngpus >= 2:
                if not isinstance(model[i], nn.DataParallel):
                    model[i] = torch.nn.DataParallel(model[i], gpu_list).cuda()
            else:
                model[i] = model[i].cuda()
    else:
        if ngpus >= 2:
            if not isinstance(model, nn.DataParallel):
                model = torch.nn.DataParallel(model, gpu_list).cuda()
        else:
            model = model.cuda()
    return model


"""def getweights(layer, epoch_id, block_id, layer_id, log_writer):
    if isinstance(layer, nn.Conv2d):
        weights = layer.weight.data.cpu().numpy()
        weights_view = weights.reshape(weights.size)
        log_writer(input_data=weights_view, block_id=block_id, layer_id=layer_id, epoch_id=epoch_id)"""


single_train_time = 0
single_test_time = 0
single_train_iters = 0
single_test_iters = 0


def getlearningrate(epoch, opt):
    # update lr
    lr = opt.LR
    if opt.lrPolicy == "multistep":
        if epoch + 1.0 > opt.nEpochs * opt.ratio[1]:  # 0.6 or 0.8
            lr = opt.LR * 0.01
        elif epoch + 1.0 > opt.nEpochs * opt.ratio[0]:  # 0.4 or 0.6
            lr = opt.LR * 0.1
    elif opt.lrPolicy == "linear":
        k = (0.001-opt.LR)/math.ceil(opt.nEpochs/2.0)
        lr = k*math.ceil((epoch+1)/opt.step)+opt.LR
    elif opt.lrPolicy == "exp":
        power = math.floor((epoch+1)/opt.step)
        lr = lr*math.pow(opt.gamma, power)
    elif opt.lrPolicy == "fixed":
        lr = opt.LR
    else:
        assert False, "invalid lr policy"

    return lr


def computetencrop(outputs, labels):
    output_size = outputs.size()
    outputs = outputs.view(output_size[0]/10, 10, output_size[1])
    outputs = outputs.sum(1).squeeze(1)
    # compute top1
    _, pred = outputs.topk(1, 1, True, True)
    pred = pred.t()
    top1_count = pred.eq(labels.data.view(1, -1).expand_as(pred)).view(-1).float().sum(0)
    top1_error = 100.0 - 100.0 * top1_count / labels.size(0)
    top1_error = float(top1_error.cpu().numpy())

    # compute top5
    _, pred = outputs.topk(5, 1, True, True)
    pred = pred.t()
    top5_count = pred.eq(labels.data.view(1, -1).expand_as(pred)).view(-1).float().sum(0)
    top5_error = 100.0 - 100.0 * top5_count / labels.size(0)
    top5_error = float(top5_error.cpu().numpy())
    return top1_error, 0, top5_error


def computeresult(outputs, labels, loss, top5_flag=False):
    if isinstance(outputs, list):
        top1_loss = []
        top1_error = []
        top5_error = []
        for i in range(len(outputs)):
            # get index of the max log-probability
            predicted = outputs[i].data.max(1)[1]
            top1_count = predicted.ne(labels.data).cpu().sum()
            top1_error.append(100.0*top1_count/labels.size(0))
            top1_loss.append(loss[i].data[0])
            if top5_flag:
                _, pred = outputs[i].data.topk(5, 1, True, True)
                pred = pred.t()
                top5_count = pred.eq(labels.data.view(1, -1).expand_as(pred)).view(-1).float().sum(0)
                single_top5 = 100.0 - 100.0 * top5_count / labels.size(0)
                single_top5 = float(single_top5.cpu().numpy())
                top5_error.append(single_top5)

    else:
        # get index of the max log-probability
        predicted = outputs.data.max(1)[1]
        top1_count = predicted.ne(labels.data).cpu().sum()
        top1_error = 100.0*top1_count/labels.size(0)
        top1_loss = loss.data[0]
        top5_error = 100.0
        if top5_flag:
            _, pred = outputs.data.topk(5, 1, True, True)
            pred = pred.t()
            top5_count = pred.eq(labels.data.view(1, -1).expand_as(pred)).view(-1).float().sum(0)
            top5_error = 100.0 - 100.0 * top5_count/labels.size(0)
            top5_error = float(top5_error.cpu().numpy())

    if top5_flag:
        return top1_error, top1_loss, top5_error
    else:
        return top1_error, top1_loss


def printresult(epoch, nEpochs, count, iters, lr, data_time, iter_time, error, loss, top5error=None, mode="Train"):
    global single_train_time, single_test_time
    global single_train_iters, single_test_iters

    # log_str = ">>> %s [%.3d|%.3d], Iter[%.3d|%.3d], LR:%.4f, DataTime: %.4f, IterTime: %.4f" \
    #           % (mode, epoch + 1, nEpochs, count, iters, lr, data_time, iter_time)

    log_str = colored(">>> %s: "% mode, "white") + colored("[%.3d|%.3d], "%(epoch + 1, nEpochs), "magenta") \
              + "Iter: " + colored("[%.3d|%.3d], " % (count, iters), "magenta") \
              + "LR: " + colored("%.4f, "%lr, "magenta") \
              + "DataTime: " + colored("%.4f, " % data_time, "blue") \
              + "IterTime: " + colored("%.4f, " % iter_time, "blue")
    if isinstance(error, list):
        for i in range(len(error)):
            # log_str += ", Error_%d: %.4f, Loss_%d: %.4f" % (i, error[i], i, loss[i])
            log_str += "Error_%d: " % i + colored("%.4f, " % error[i], "cyan") \
                       + "Loss_%d: " % i + colored("%.4f, " % loss[i], "cyan")
    else:
        # log_str += ", Error: %.4f, Loss: %.4f" % (error, loss)
        log_str += "Error: " + colored("%.4f, " % error, "cyan") \
                   + "Loss: " + colored("%.4f, " % loss, "cyan")

    if top5error is not None:
        if isinstance(top5error, list):
            for i in range(len(top5error)):
                # log_str += ", Top5_Error_%d: %.4f" % (i, top5error[i])
                log_str += " Top5_Error_%d:" % i + colored("%.4f, " % top5error[i], "cyan")
        else:
            # log_str += ", Top5_Error: %.4f" % top5error
            log_str += "Top5_Error: " + colored("%.4f, " % top5error, "cyan")

    # compute cost time
    if mode == "Train":
        single_train_time = single_train_time*0.95 + 0.05*(data_time+iter_time)
        # single_train_time = data_time + iter_time
        single_train_iters = iters
        train_left_iter = single_train_iters-count+(nEpochs-epoch-1)*single_train_iters
        # print "train_left_iters", train_left_iter
        test_left_iter = (nEpochs-epoch)*single_test_iters
    else:
        single_test_time = single_test_time * 0.95 + 0.05 * (data_time + iter_time)
        # single_test_time = data_time+iter_time
        single_test_iters = iters
        train_left_iter = (nEpochs - epoch-1) * single_train_iters
        test_left_iter = single_test_iters - count + (nEpochs - epoch-1) * single_test_iters

    left_time = single_train_time*train_left_iter+single_test_time*test_left_iter
    total_time = (single_train_time*single_train_iters+single_test_time*single_test_iters)*nEpochs
    # time_str = ",Total Time: %s, Remaining Time: %s" % (str(datetime.timedelta(seconds=total_time)),
    #                                                     str(datetime.timedelta(seconds=left_time)))
    time_str = "Total Time: " + colored("%s, " % str(datetime.timedelta(seconds=total_time)), "red") \
               + "Remaining Time: " + colored("%s" % str(datetime.timedelta(seconds=left_time)), "red")

    print log_str+time_str
    return total_time, left_time



def list2sequential(model):
    if isinstance(model, list):
        model = nn.Sequential(*model)
    return model


def paramscount(model):
    # counting numbers of parameters
    params_size = 0
    model_list = MD.model2list(model)
    for i in range(len(model_list)):
        for params in model_list[i].parameters():
            params_numpy = params.data.cpu().numpy()
            params_size += params_numpy.size

    print "number of parameters is: ", params_size
    return params_size


def getallweights(model):
    # get weights from model
    model_list = MD.model2list(model)
    weight_np = None
    for i in range(len(model_list)):
        model_state_dict = model_list[i].state_dict()
        for k, d in model_state_dict.items():
            k_split = k.split(".")
            if k_split[-1] == "weight":
                d_np = d.cpu().numpy()
                d_np = d_np.reshape(d_np.size, 1)
                if weight_np is None:
                    weight_np = d_np
                else:
                    weight_np = np.row_stack((weight_np, d_np))
    return weight_np
