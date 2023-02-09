from torch.utils.data import DataLoader, Subset
"""
    trans = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize((224,512)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
"""
def split_train_val(data, bsize, workers=8, split=0.9):
    """ Split a dataset in train and val. It returns a train and val tuple"""
    
    train_split = int(len(data) * split)
    train_sbs = Subset(data, range(train_split))
    val_sbs = Subset(data, range(train_split, len(data)))
    train_loader = DataLoader(train_sbs, batch_size=bsize, shuffle=True, num_workers=workers)
    val_loader = DataLoader(val_sbs, batch_size=bsize, shuffle=True, num_workers=workers)

    return train_loader, val_loader

def split_train_val_test(data, batch_size, splits=(0.8, 0.1, 0.1), nworkers=8):
    train, val, test = splits
    
    num_examples_train = int(len(data) * train)
    num_examples_val = int(len(data) * val)

    train_sbs = Subset(data, range(num_examples_train))
    val_sbs = Subset(data, range(num_examples_train, num_examples_train+num_examples_val))
    test_sbs = Subset(data, range(num_examples_train+num_examples_val, len(data)))
    
    assert len(test_sbs) ==  len(data) - num_examples_train - num_examples_val, "len subset is wrong"


    train_loader = DataLoader(train_sbs, batch_size=batch_size, shuffle=True, num_workers=nworkers)
    val_loader = DataLoader(val_sbs, batch_size=batch_size, shuffle=True, num_workers=nworkers)
    test_loader = DataLoader(val_sbs, batch_size=batch_size, shuffle=True, num_workers=nworkers)

    return train_loader, val_loader, test_loader

