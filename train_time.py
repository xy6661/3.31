# import time
# from data import create_dataset
# from models import create_model
# from util.visualizer import Visualizer
# from options.train_options import TrainOptions
# from tqdm import tqdm  # <-- 1. 导入 tqdm 库
#
# if __name__ == '__main__':
#     opt = TrainOptions().parse()
#     dataset = create_dataset(opt)
#     dataset_size = len(dataset)
#     print('The number of training samples = %d' % dataset_size)
#     model = create_model(opt)
#     model.setup(opt)
#     visualizer = Visualizer(opt)
#     total_iters = 0
#     total_batch_iters = 0
#
#     # 外层 epoch 循环保持不变
#     for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
#         epoch_start_time = time.time()
#         epoch_iter = 0
#
#         # --- 2. 创建 tqdm 进度条对象 ---
#         # desc 参数用于在进度条前显示当前是第几个 epoch
#         # len(dataset) 告诉 tqdm 这个 epoch 的总迭代次数是多少
#         # unit="batch" 会让进度条的单位显示为 "batch"
#         pbar = tqdm(dataset, desc=f"Epoch {epoch}/{opt.n_epochs + opt.n_epochs_decay}", unit="batch")
#
#         # --- 3. 在进度条对象上进行迭代 ---
#         for i, data in enumerate(pbar):  # 每次加载bachsize张图片的时候i就会增加一次
#             iter_start_time = time.time()
#             total_iters += opt.batch_size
#             epoch_iter += opt.batch_size
#             total_batch_iters += 1
#             model.set_input(data)
#             model.optimize_parameters()
#
#             # 内部的所有逻辑完全保持不变
#             if total_batch_iters % opt.display_freq == 0:
#                 visuals = model.get_current_visuals()
#                 losses = model.get_current_losses()
#                 visualizer.display_current_results(visuals, epoch, total_iters % opt.update_html_freq == 0)
#                 t_comp = (time.time() - iter_start_time) / opt.batch_size
#
#                 # 保持原始的打印损失函数调用
#                 visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp)
#
#                 # --- 4. (可选但推荐) 将损失信息也附加到进度条的末尾 ---
#                 pbar.set_postfix(losses)
#
#                 if opt.display_id > 0:
#                     visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)
#
#             if total_batch_iters % opt.save_latest_freq == 0:
#                 # 原始的打印保存信息逻辑保持不变
#                 print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
#                 save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
#                 model.save_networks(save_suffix)
#
#         # epoch 结束后的所有逻辑也完全保持不变
#         if epoch % opt.save_epoch_freq == 0:
#             print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
#             model.save_networks('latest')
#             model.save_networks(epoch)
#
#         print('End of epoch %d / %d \t Time Taken: %d sec' %
#               (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
#         model.update_learning_rate()

import time
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from options.train_options import TrainOptions
from tqdm import tqdm  # <-- 1. 导入 tqdm 库

if __name__ == '__main__':
    opt = TrainOptions().parse()
    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    print('The number of training samples = %d' % dataset_size)
    model = create_model(opt)
    model.setup(opt)
    visualizer = Visualizer(opt)
    total_iters = 0
    total_batch_iters = 0

    # 原始的 epoch 循环保持不变
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()
        epoch_iter = 0

        # --- 2. 在 for 循环外创建 tqdm 进度条 ---
        # 我们手动创建一个进度条实例，总数(total)设置为该 epoch 的总样本数 (dataset_size)。
        # 这样进度条会以处理的图片数量为单位进行更新。
        # desc 参数会在进度条前显示当前 epoch 的信息。
        # unit='img' 将进度条的单位设置为 'img'，更直观。
        with tqdm(total=dataset_size, desc=f"Epoch {epoch}/{opt.n_epochs + opt.n_epochs_decay}", unit='img') as pbar:
            # --- 3. 循环遍历原始的 dataset ---
            # 这里的循环目标是原始的 dataset，而不是 tqdm 对象。
            for i, data in enumerate(dataset):  # 每次加载bachsize张图片的时候i就会增加一次
                # ==========================================================
                # 以下所有代码均为您的原始逻辑，完全不做任何改动
                # ==========================================================
                iter_start_time = time.time()
                total_iters += opt.batch_size
                epoch_iter += opt.batch_size
                total_batch_iters += 1
                model.set_input(data)
                model.optimize_parameters()
                if total_batch_iters % opt.display_freq == 0:
                    visuals = model.get_current_visuals()  # 此处参数应该有self和is_validation，但是python会自动填写self也就是调用的这个model会自动调用这个self
                    losses = model.get_current_losses()
                    visualizer.display_current_results(visuals, epoch, total_iters % opt.update_html_freq == 0)
                    t_comp = (time.time() - iter_start_time) / opt.batch_size
                    visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp)
                    if opt.display_id > 0:
                        visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

                    # --- 4. (附加功能) 将损失信息也显示在进度条右侧 ---
                    # 这行代码不会影响您原有的打印逻辑，只是让进度条信息更丰富。
                    pbar.set_postfix(losses)

                if total_batch_iters % opt.save_latest_freq == 0:
                    print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                    save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                    model.save_networks(save_suffix)

                # --- 5. 手动更新进度条的进度 ---
                # 在处理完一个批次后，调用 update() 方法来更新进度条。
                # 更新的量就是批次大小 (opt.batch_size)。
                pbar.update(opt.batch_size)

        # ==========================================================
        # epoch 结束后的逻辑也完全保持不变
        # ==========================================================
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)
        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        model.update_learning_rate()