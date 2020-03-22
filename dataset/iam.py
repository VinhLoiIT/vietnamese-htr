from torch.utils.data import Dataset

class IAM(Dataset):
    def __init__(self, partition_split, image_transform=None):
        self.image_folder = './data/IAM/words'
        self.annotation_file = './data/IAM/words.txt'
        with open (partition_split) as f:
            partition_folder = f.readlines()
        self.partition_folder = [x.strip() for x in partition_folder]
        self.image_transform = image_transform

        self.content = [] # [[path, label]*N]
        with open (self.annotation_file) as f:
            for line in f:
                if not line or line.startswith('#'): # comment in txt file
                    continue
                line_split = line.strip().split(' ')
                assert len(line_split) >= 9
                status = line_split[1]
                if status == 'err': # er: segmentation of word can be bad
                    continue

                file_name_split = line_split[0].split('-')
                label_dir = file_name_split[0]
                sub_label_dir = '{}-{}'.format(file_name_split[0], file_name_split[1])
                fn = '{}.png'.format(line_split[0])
                img_path = os.path.join(label_dir, sub_label_dir, fn)

                gt_text = ' '.join(line_split[8:]) # Word, but contains spaces

                if sub_label_dir in self.partition_folder:
                    self.content.append([img_path, gt_text])

    def __len__(self):
        return len(self.content)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_folder, self.content[idx][0])
        image = Image.open(image_path).convert('L')

        if self.image_transform:
            image = self.image_transform(image)

        label = self.content[idx][1]
        label = [SOS_CHAR] + list(label) + [EOS_CHAR]

        return image, label