# 构建和运行多智能体服务基础调度及执行框架

[English](README.md) | [中文](README_zh.md)

## 简介
多智能体服务基础调度及执行框架包含以下四部分，能够支撑服务化（容器化部署）多智能体系统的运行，协同，上下文管理和多模态文件存储。MDOC和MAR多智能体服务系统均以此为基础设施快速接入、搭建和运行。
1. 关系型数据库MySQL：管理和存储任务执行状态和执行图地址
2. 调度器及文件存储Java服务center、taskScheduling和storage：多智能体调度器和文件存储服务封装
3. 多模态对象存储MinIO：存储执行图graph.json及多智能体产生的所有多模态文件
4. 多智能体服务注册与发现中心Nacos：管理和维护多智能体服务动态发现和协同寻址

## 构建方式
该多智能体服务基础调度及执行框架以Docker形式构建和运行，具体使用的运行环境及版本：
该镜像基于 **Ubuntu:18.04**，集成并运行以下组件：**Java 1.8**、**MySQL 5.7、Nacos 2.4.3、MinIO、Eureka（center.jar）、storage.jar 和 taskScheduling.jar**。三个 JAR 均使用 `exp` Spring Profile 启动；Nacos 以 standalone 模式运行。MinIO 在镜像构建时从官方源码编译最新社区版。

### 下载上述所需大文件
https://www.dropbox.com/scl/fi/654n537pu02rgb7tjj454/scheduler_bf.zip?rlkey=tgn6jc32puy2quvbixhtvydvl&st=ivo9nw0k&dl=0
下载完成后放置于benchmark/scheduler文件夹下

### 构建docker镜像
在本目录执行：

```bash
docker build --platform linux/amd64 -t microarcl-scheduler .
# 可选使用docker命名卷挂载
# docker volume create microarcl-mysql
# docker volume create microarcl-minio

# 当前指令使用绑定挂载方式
docker run -d --platform linux/amd64 --name microarcl-scheduler --restart unless-stopped \
  -e HOST_IP=192.168.0.109 \
  -v /your_local_dir/mysql:/var/lib/mysql \
  -v /your_local_dir/minio:/data/minio \
  -p 3306:3306 -p 8848:8848 -p 9848:9848 \
  -p 9000:9000 -p 9001:9001 -p 32196:32196 \
  -p 12104:12104 -p 35696:35696 -p 38001:38001 \
  microarcl-scheduler
```

`HOST_IP` 应为容器可以被访问的宿主机地址。启动脚本会把 `minio.agent.network.com`、`db.agent.network.com`、`taskscheduling.agent.network.com` 和 `center.agent.network.com` 映射到此地址。容器外的各台K8S节点也需要自行配置这些域名的解析，使K8S中各智能体服务pod能够访问该容器以进行多智能体调度和文件存储。

查看启动日志：

```bash
docker logs -f microarcl-scheduler
```
