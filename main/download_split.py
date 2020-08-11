#%%
import os
path = r'C:\Users\DEEP\PycharmProjects\NCFile\main\link'
file = open(r'C:\Users\DEEP\PycharmProjects\NCFile\main\l2.txt')

#%%
link = file.readlines()

#%%
l1 = link[:330]
l2 = link[330: 832]
l3 = link[832:]

#%%
f1 = open(os.path.join(path, 'l1.txt'), mode='w')
f2 = open(os.path.join(path, 'l2.txt'), mode='w')
f3 = open(os.path.join(path, 'l3.txt'), mode='w')

f1.write(''.join(l1))
f2.write(''.join(l2))
f3.write(''.join(l3))

f1.close()
f2.close()
f3.close()
#%%
c = 0
for i, l in enumerate(link):
    if l.strip().endswith('.nc'):
        c += 1
