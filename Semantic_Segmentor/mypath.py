class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'cityscapes':
            return r'<root_path>/Cityscapes/'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError("undefined dataset {}.".format(dataset))
