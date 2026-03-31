# import torch.multiprocessing
# # 必须在其他 import 之前尽早设置
# torch.multiprocessing.set_sharing_strategy('file_system')
# import time
# import itertools  # 导入 itertools 库
# from tqdm import tqdm  # 确保 tqdm 已导入
# import sys  # 导入 sys 库，用于健壮性检查
# from data import create_dataset
# from models import create_model
# from util.visualizer import Visualizer
# from options.train_options import TrainOptions
#
# if __name__ == '__main__':
#     opt = TrainOptions().parse()
#     dataset = create_dataset(opt)
#     dataset_size = len(dataset)
#     print('训练样本的数量 = %d' % dataset_size)
#     if dataset_size ==0:    #【二次修正】健壮性检查
#         print(("错误：数据集中没有找到任何图片，程序退出"))
#         sys.exit()
#     model = create_model(opt)
#     model.setup(opt)
#     visualizer = Visualizer(opt)
#
#     # --- 修改开始 ---
#     total_iterations_target = 160000    # 1. 定义总迭代次数
#     data_iterator = iter(itertools.cycle(dataset))# 2. 将 dataset 包装成一个无限循环的迭代器
#                                                   # 这样我们就不需要手动处理轮次结束和重新开始的问题了
#     # 3. 使用 tqdm 创建主训练循环和进度条
#     # tqdm 会自动处理进度条、已用时间、剩余时间和迭代速度的显示
#
#
#     """for total_batch_iters in tqdm(range(1, total_iterations_target + 1), desc="训练进度"):
#         上面的代码被下面两行取代，以后想要换回需要逐一缩进"""
#     with tqdm(range(1, total_iterations_target + 1), desc="训练进度") as pbar:
#         for total_batch_iters in pbar:
#             data = next(data_iterator)  # 从无限迭代器中获取下一批数据
#             iter_start_time = time.time()
#
#             model.set_input(data)  # 这个非常关键，也就是这个玩意传递了处理后的内容/风格图片
#             model.optimize_parameters() #这个方法直接会调用adaattn_model的optimize_parameters方法
#
#             # 4. 更新进度条上的损失信息，并记录到日志
#             if total_batch_iters % opt.display_freq == 0:
#                 losses = model.get_current_losses()
#
#                 """# 使用 tqdm 的 set_postfix 方法，将损失信息动态显示在进度条上
#                 tqdm.set_postfix(tqdm.get_postfix(), **losses)"""
#                 pbar.set_postfix(**losses) # 【修正】使用 pbar 实例来更新后缀，更健壮
#
#                 # 仍然调用 print_current_losses 来将日志写入文件
#                 t_comp = (time.time() - iter_start_time) / opt.batch_size
#
#                 """# 注意：这里的 visualizer.print_current_losses 主要是为了写 loss_log.txt 文件
#                 visualizer.print_current_losses(int(total_batch_iters / dataset_size) + 1, total_batch_iters, losses,
#                                                 t_comp)"""
#
#                 current_epoch = (total_batch_iters - 1) // dataset_size + 1# 【修正】使用更精确的轮次计算公式
#                 visualizer.print_current_losses(current_epoch, total_batch_iters, losses, t_comp)
#
#
#                 # 可视化部分（如果使用 visdom）
#
#                 # --- 修改后的代码块 ---
#                 # 1. 检查是否需要进行 可视化 或 本地保存
#                 should_save_html = not opt.no_html and total_batch_iters % opt.update_html_freq == 0
#                 should_display_visdom = opt.display_id > 0
#                 if should_save_html or should_display_visdom > 0:#opt.display_id
#                     visuals = model.get_current_visuals()
#                     visualizer.display_current_results(visuals, current_epoch,
#                                                        total_batch_iters % opt.update_html_freq == 0)
#                     # visualizer.plot_current_losses(current_epoch,
#                     #                                float(total_batch_iters % dataset_size) / dataset_size, losses)
#                 # 2. 仅当 Visdom 启用时，才绘制损失曲线
#                 if should_display_visdom:
#                     visualizer.plot_current_losses(current_epoch,
#                                                    float(total_batch_iters % dataset_size) / dataset_size, losses)
#                 # --- 修改结束 ---
#
#             # 5. 此处保持不变--保存模型的逻辑
#             if total_batch_iters % opt.save_latest_freq == 0:
#                 """print(f' (迭代 {total_batch_iters}) 正在保存最新模型...')"""
#                 tqdm.write(f' (迭代 {total_batch_iters}) 正在保存最新模型...')  # 【修正】使用 tqdm.write 替代 print，避免打乱进度条
#
#                 save_suffix = 'iter_%d' % total_batch_iters if opt.save_by_iter else 'latest'
#                 model.save_networks(save_suffix)
#
#             # 6. 更新学习率的逻辑 (按等效轮次)
#             # 每当处理的批次数是数据集大小的整数倍时，就认为一个“轮次”已过
#             if total_batch_iters % dataset_size == 0:
#                 epoch = total_batch_iters // dataset_size
#                 """print(f' (迭代 {total_batch_iters}) 完成等效轮次 {epoch}, 更新学习率...')"""
#                 tqdm.write(f' (迭代 {total_batch_iters}) 完成等效轮次 {epoch}, 更新学习率...')  # 【修正】使用 tqdm.write
#
#                 model.update_learning_rate()
#
#                 # 按轮次频率保存模型
#                 if epoch % opt.save_epoch_freq == 0:
#                     """print(f'在等效轮次 {epoch} 结束时保存模型...')"""
#                     tqdm.write(f'在等效轮次 {epoch} 结束时保存模型...')  # 【修正】使用 tqdm.write
#
#                     model.save_networks('latest')
#                     model.save_networks(epoch)
#
#
#     print('训练在 %d 次迭代后完成。' % total_iterations_target)
#
#     # --- 修改结束 ---

import time
import itertools
from tqdm import tqdm
import sys
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from options.train_options import TrainOptions

if __name__ == '__main__':
    opt = TrainOptions().parse()
    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    print('训练样本的数量 = %d' % dataset_size)
    if dataset_size == 0:
        print("错误：数据集中没有找到任何图片，程序退出")
        sys.exit()

    model = create_model(opt)
    model.setup(opt)
    visualizer = Visualizer(opt)

    # --- 优化开始 ---

    # 1. 定义总的训练目标迭代次数
    total_iterations_target = 160000

    # 2. 定义一个“重置周期”，每隔这么多此迭代就重新创建一次 DataLoader
    #    这可以有效地防止因子进程内存泄漏导致的长时间训练崩溃问题。
    #    如果您的服务器内存较小，可以适当调低此值 (例如 5000 或 8000)。
    ITERATIONS_PER_RESET = 2000

    print('开始训练...')

    # 3. 创建一个总的 tqdm 进度条，手动控制更新
    with tqdm(total=total_iterations_target, desc="训练进度") as pbar:

        total_batch_iters = 0
        # 4. 主循环，直到达到总迭代目标
        while total_batch_iters < total_iterations_target:

            # 5. 【核心优化点】在每个重置周期开始时，重新创建数据迭代器
            #    这会重置 DataLoader 和它所有的 worker 子进程，从而释放累积的内存。
            tqdm.write(f'\n(迭代 {total_batch_iters}) 正在重置数据加载器以释放内存...')
            data_iterator = iter(itertools.cycle(dataset))

            # 6. 计算本次内部循环需要运行的迭代次数
            iterations_this_cycle = min(ITERATIONS_PER_RESET, total_iterations_target - total_batch_iters)

            # 7. 内部循环，执行一个重置周期内的训练
            for _ in range(iterations_this_cycle):

                # 从无限迭代器中获取下一批数据
                data = next(data_iterator)
                iter_start_time = time.time()

                # 模型训练步骤
                model.set_input(data)
                model.optimize_parameters()

                # 更新总迭代计数器
                total_batch_iters += 1
                # 手动更新总进度条
                pbar.update(1)

                # --- 日志、可视化和模型保存逻辑 ---

                # 更新进度条上的损失信息，并记录到日志
                if total_batch_iters % opt.display_freq == 0:
                    losses = model.get_current_losses()
                    pbar.set_postfix(**losses)

                    t_comp = (time.time() - iter_start_time) / opt.batch_size
                    current_epoch = (total_batch_iters - 1) // dataset_size + 1
                    visualizer.print_current_losses(current_epoch, total_batch_iters, losses, t_comp)

                    # 图片保存与 Visdom 可视化解耦逻辑
                    should_save_html = not opt.no_html and total_batch_iters % opt.update_html_freq == 0
                    should_display_visdom = opt.display_id > 0

                    if should_save_html or should_display_visdom:
                        visuals = model.get_current_visuals()
                        visualizer.display_current_results(visuals, current_epoch, save_result=should_save_html)

                    if should_display_visdom:
                        visualizer.plot_current_losses(current_epoch, float(total_batch_iters % dataset_size) / dataset_size, losses)

                # 保存最新模型
                if total_batch_iters % opt.save_latest_freq == 0:
                    tqdm.write(f' (迭代 {total_batch_iters}) 正在保存最新模型...')
                    save_suffix = 'iter_%d' % total_batch_iters if opt.save_by_iter else 'latest'
                    model.save_networks(save_suffix)

                # 按等效轮次更新学习率和保存模型
                if total_batch_iters % dataset_size == 0:
                    epoch = total_batch_iters // dataset_size
                    tqdm.write(f' (迭代 {total_batch_iters}) 完成等效轮次 {epoch}, 更新学习率...')
                    model.update_learning_rate()

                    if epoch % opt.save_epoch_freq == 0:
                        tqdm.write(f'在等效轮次 {epoch} 结束时保存模型...')
                        model.save_networks('latest')
                        model.save_networks(epoch)

    print('训练在 %d 次迭代后完成。' % total_iterations_target)
    # --- 优化结束 ---