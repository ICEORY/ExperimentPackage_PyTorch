import torch
import os
import utils

# Note:
'''
# save and load entire model
torch.save(model, "model.pkl")
model = torch.load("model.pkl")
# save and load only the model parameters(recommended)
torch.save(model.state_dict(), "params.pkl")
model.load_state_dict(torch.load("params.pkl"))
'''


class CheckPoint(object):
    def __init__(self, opt):
        self.resume = opt.resume
        self.resumeEpoch = opt.resumeEpoch
        self.retrain = opt.retrain
        self.save_path = opt.save_path+"model/"
        self.check_point_params = {'model': None,
                                   'opts': None,
                                   'resume_epoch': None}

    def retrainmodel(self):
        if os.path.isfile(self.retrain):
            print "|===>Retrain model from:", self.retrain
            retrain_data = torch.load(self.retrain)
            self.check_point_params['model'] = retrain_data['model']
            return self.check_point_params
        else:
            assert False, "file not exits"

    def resumemodel(self):
        if os.path.isfile(self.resume):
            print "|===>Resume check point from:", self.resume
            self.check_point_params = torch.load(self.resume)
            if self.resumeEpoch != 0:
                self.check_point_params['resume_epoch'] = self.resumeEpoch
            return self.check_point_params
        else:
            assert False, "file not exits"

    def savemodel(self, epoch=None, model=None, opts=None, best_flag=False):
        # Note: if we add hook to the grad by using register_hook(hook), then the hook function can not be saved
        # so we need to save state_dict() only. Although save state dictionary is recommended, I still want to save
        # the whole model as it can save the structure of network too, thus we do not need to create a new network
        # next time.

        # model = utils.list2sequential(model).state_dict()
        # opts = opts.state_dict()

        if not os.path.isdir(self.save_path):
            os.mkdir(self.save_path)
        # self.check_point_params['model'] = utils.list2sequential(model).state_dict()
        self.check_point_params['model'] = model
        self.check_point_params['opts'] = opts
        self.check_point_params['resume_epoch'] = epoch

        torch.save(self.check_point_params, self.save_path+"checkpoint.pkl")
        if best_flag:
            # best_model = {'model': utils.list2sequential(model).state_dict()}
            best_model = {'model': model}
            torch.save(best_model, self.save_path+"best_model.pkl")
