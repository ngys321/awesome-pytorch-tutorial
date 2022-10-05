#
# 특징1 : train_set, test_set 분리되지 않은 경우임.
#        train_set, test_set 으로 분리되지 않은 경우, split 하여 train_set, test_set 각각 만들어서 사용
# 특징2 : data 가 RGB image 와 depth image 의 pair 로 이루어져 있음.
#        
import torch
train_ratio = 0.8 # train_ratio : train_set 비율

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
    
        from collections import OrderedDict
        
        #label indexing 용 ordered dictionary
        self.ordered_dic_for_label = OrderedDict()
        #img data indexing 용 ordered dictionary
        self.ordered_dic_for_rgb = OrderedDict()
        #depth data indexing 용 ordered dictionary
        self.ordered_dic_for_depth = OrderedDict()
        
        #label indexing##################################################################
        from os.path import isfile, join                                                #
        import json                                                                     #
        f = open(join(img_dir,"GroundTruth/GroundTruth_All_388_Images.json"))# 파일 열음  #
        data = json.load(f)                                                             #
                                                                                        #
        # (index, label) -> ordered_dic_for_label 에 넣기                                #
        for index, (key, value) in enumerate(data["Measurements"].items()):             #
            self.ordered_dic_for_label[int(index)] = value['FreshWeightShoot']          #
        f.close()# 파일 닫음                                                              #
        #################################################################################
        
        #rgb data indexing#############################################################################
        from os.path import isfile, join                                                              #
        import json                                                                                   #
        f = open(join(img_dir,"GroundTruth/GroundTruth_All_388_Images.json"))# 파일 열음                #
        data = json.load(f)                                                                           #
        # (index, rgb tensor) -> ordered_dic_for_rgb 에 넣기                                           #
        import matplotlib.image as mpimg                                                              #
        import torch                                                                                  #
        for index, (key, value) in enumerate(data["Measurements"].items()):                           #
            img = mpimg.imread(join(img_dir,"RGBImages/", str(value['RGB_Image'])))                   #
            tensor = torch.from_numpy(img)                                                            #
            print("img data size : ",tensor.size(), "index : ", index) # torch.Size([1080, 1920, 3])  #
            #print(tensor)                                                                            #
            self.ordered_dic_for_rgb[int(index)] = tensor                                             #
        f.close()# 파일 닫음                                                                            #
        ###############################################################################################
        
        #depth data indexing###########################################################################
        from os.path import isfile, join                                                              #
        import json                                                                                   #
        f = open(join(img_dir,"GroundTruth/GroundTruth_All_388_Images.json"))# 파일 열음                #
        data = json.load(f)                                                                           #
        # (index, depth tensor) -> ordered_dic_for_depth 에 넣기                                       #
        import matplotlib.image as mpimg                                                              #
        import torch                                                                                  #
        for index, (key, value) in enumerate(data["Measurements"].items()):                           #
            img = mpimg.imread(join(img_dir,"/DepthImages/", str(value['Depth_Information'])))        #
            tensor = torch.from_numpy(img)                                                            #
            print("depth data size : ",tensor.size(), "index : ", index) # torch.Size([1080, 1920])   #
            #print(tensor)                                                                            #
            self.ordered_dic_for_depth[int(index)] = tensor                                           #
                                                                                                      #
        f.close()# 파일 닫음                                                                            #
        ###############################################################################################

        #print(len(self.ordered_dic_for_label))
        #print(len(self.ordered_dic_for_rgb))
        #print(len(self.ordered_dic_for_depth))
        # 위 3개 모두 387로 같은 값
        
    def __getitem__(self, index):
        label = ordered_dic_for_label[index]
        rgb = ordered_dic_for_rgb[index]
        depth = ordered_dic_for_depth[index]
        
        return (rgb, depth), label  #(rgb image, depth image) tuple 을 하나의 데이터로 return 함. 
                                    #나중에 dataloader 사용할땐, x[0], x[1] 로 indexing 해서 사용하는 것은 추천.
    
    def __len__(self):
        
        assert len(self.ordered_dic_for_label) != len(self.ordered_dic_for_rgb) \
                or len(self.ordered_dic_for_rgb) != len(self.ordered_dic_for_depth)\
                ,"ordered_dics have different lengths"
        return len(self.ordered_dic)
    
    
full_dataset = CustomDataset(img_dir="./lettuce_dataset")

train_size = int(train_ratio * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])


train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=4,
                                          shuffle=True)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=4,
                                          shuffle=True)