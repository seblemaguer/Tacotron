import os

for file in os.listdir("4books_batch2"):
  info = file.split("_")
  id = info[0]
  new_id = int(id) + 100

  new_name = new_id+"_"+info[1]+"_"+info[2]+"_"+info[3]

  print(f'mv {file} {new_name}')

  
