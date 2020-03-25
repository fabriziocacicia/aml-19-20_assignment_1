import matplotlib.pyplot as plt

import rpc_module

with open('model.txt') as fp:
    model_images = fp.readlines()
model_images = [x.strip() for x in model_images]

with open('query.txt') as fp:
    query_images = fp.readlines()
query_images = [x.strip() for x in query_images]

num_bins = 20

plt.figure(8)
rpc_module.compare_dist_rpc(model_images, query_images, ['chi2', 'intersect', 'l2'], 'rg', num_bins, ['r', 'g', 'b'])
plt.title('RG histograms')
plt.show()

plt.figure(9)
rpc_module.compare_dist_rpc(model_images, query_images, ['chi2', 'intersect', 'l2'], 'rgb', num_bins // 2,
                            ['r', 'g', 'b'])
plt.title('RGB histograms')
plt.show()

plt.figure(10)
rpc_module.compare_dist_rpc(model_images, query_images, ['chi2', 'intersect', 'l2'], 'dxdy', num_bins, ['r', 'g', 'b'])
plt.title('dx/dy histograms')
plt.show()
