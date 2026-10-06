# Build and Run the Multi-Agent Service Scheduling and Execution Framework

[English](README.md) | [中文](README_zh.md)

## Overview

The multi-agent service scheduling and execution framework consists of four parts. It supports the operation, coordination, context management, and multimodal file storage of service-based (containerized) multi-agent systems. Both the MDOC and MAR multi-agent service systems use it as infrastructure for rapid integration, deployment, and operation.

1. **MySQL relational database:** Manages and stores task execution states and execution graph locations.
2. **Java scheduling and file-storage services (`center`, `taskScheduling`, and `storage`):** Provide multi-agent scheduling and file-storage services.
3. **MinIO multimodal object storage:** Stores the execution graph (`graph.json`) and all multimodal files produced by the agents.
4. **Nacos service registry and discovery center:** Manages dynamic discovery and coordinated addressing for multi-agent services.

## Build and Run

This framework is built and run with Docker. The image is based on **Ubuntu 18.04** and includes **Java 1.8, MySQL 5.7, Nacos 2.4.3, MinIO, Eureka (`center.jar`), `storage.jar`, and `taskScheduling.jar`**. All three JARs start with the `exp` Spring profile. Nacos runs in standalone mode. During the image build, MinIO is compiled from the latest official community-edition source code.

### Download the required large files

Download the files from:

https://www.dropbox.com/scl/fi/654n537pu02rgb7tjj454/scheduler_bf.zip?rlkey=tgn6jc32puy2quvbixhtvydvl&st=ivo9nw0k&dl=0

After downloading, place them in the `benchmark/scheduler` directory.

### Build the Docker image

Run these commands from this directory:

```bash
docker build --platform linux/amd64 -t microarcl-scheduler .
# Optionally create Docker named volumes
# docker volume create microarcl-mysql
# docker volume create microarcl-minio

# The following command uses bind mounts
docker run -d --platform linux/amd64 --name microarcl-scheduler --restart unless-stopped \
  -e HOST_IP=192.168.0.109 \
  -v /your_local_dir/mysql:/var/lib/mysql \
  -v /your_local_dir/minio:/data/minio \
  -p 3306:3306 -p 8848:8848 -p 9848:9848 \
  -p 9000:9000 -p 9001:9001 -p 32196:32196 \
  -p 12104:12104 -p 35696:35696 -p 38001:38001 \
  microarcl-scheduler
```

Set `HOST_IP` to the host address through which the container can be reached. The startup script maps `minio.agent.network.com`, `db.agent.network.com`, `taskscheduling.agent.network.com`, and `center.agent.network.com` to this address. Configure DNS resolution for these domains on each K8s node outside the container as well, so agent service pods in K8s can reach the container for multi-agent scheduling and file storage.

View the startup logs:

```bash
docker logs -f microarcl-scheduler
```
