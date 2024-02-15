import os 
main_directory = "/content"
train_directory = os.path.join(main_directory,"train")
val_directory = os.path.join(main_directory,"valid")
test_directory  = os.path.join(main_directory,"test")

def remove_unique_file(dir):
  images_dir = os.path.join(dir,"images")
  labels_dir = os.path.join(dir,"labels") 
  images_name = [image.split(".")[0] for image in os.listdir(images_dir)]
  labels_name = [label.split(".")[0] for label in os.listdir(labels_dir)]
  for image_name in images_name:
    if image_name not in labels_name:
      os.remove(os.path.join(images_dir,image_name+".jpg"))
      print(f"Remove" ,os.path.join(images_dir,image_name+".jpg"))
  for label_name in labels_name:
    if label_name not in labels_name:
      os.remove(os.path.join(labels_dir,label_name+".txt"))
      print("Remove", os.path.join(labels_dir,label_name+".txt"))
  if len(os.listdir(os.path.join(dir,"labels"))) == len(os.listdir(os.path.join(dir,"images"))): 
    print("You have" ,len(os.listdir(os.path.join(dir,"labels"))) ,"images in ",dir,"after re")
  else: 
    print("you have problems")

def remove_null_labels(dir):
  print("You have" ,  len(os.listdir(os.path.join(dir,"labels"))) ,"images in ",dir,"before cleaning")
  count = 0
  for textfile in os.listdir(os.path.join(dir,"labels")):
    if os.path.getsize(os.path.join(dir,"labels",textfile)) == 0:
        count += 1
        os.remove(os.path.join(dir,"labels",textfile))
        image_name = textfile.replace('.txt', '.jpg')
        os.remove(os.path.join(dir,"images",image_name))
  print("Removed ",count,"images from", os.path.join(dir,"labels"))
  if len(os.listdir(os.path.join(dir,"labels"))) == len(os.listdir(os.path.join(dir,"images"))): 
    print("You have" ,len(os.listdir(os.path.join(dir,"labels"))) ,"images in ",dir,"after cleaning")
  else: 
    print("you have problems")

remove_unique_file(train_directory)    
remove_unique_file(val_directory)
remove_unique_file(test_directory)
remove_null_labels(train_directory)    
remove_null_labels(val_directory)
remove_null_labels(test_directory)
