DROP TABLE IF EXISTS `sys_task`;
CREATE TABLE `sys_task` (
  `id` varchar(32) NOT NULL  COMMENT '任务ID',
  `name` varchar(4096) NOT NULL  COMMENT '名称',
  `status` Integer(3) COMMENT '状态（0:新建 1:运行中 2:成功 3:失败 4:取消 5: 暂停 6: 人为协助 7: 已完成, 8:完成失败）',
  `user_id` varchar(32) COMMENT '用户ID',
  `organize_id` varchar(32) COMMENT '组织ID',
  `graph_id` varchar(32) COMMENT '执行图ID',
  `gmt_update` timestamp(0) NOT NULL  DEFAULT current_timestamp() ON UPDATE current_timestamp() COMMENT '更新时间',
  `gmt_create` timestamp(0) NOT NULL  DEFAULT current_timestamp() COMMENT '创建时间',
  PRIMARY KEY (`id`)) ENGINE=InnoDB DEFAULT CHARSET=utf8;
DROP TABLE IF EXISTS `sys_graph`;
CREATE TABLE `sys_graph` (
  `id` varchar(32) NOT NULL  COMMENT '执行图ID',
  `file_path` varchar(1024) NOT NULL  COMMENT '文件',
  `gmt_update` timestamp(0) NOT NULL  DEFAULT current_timestamp() ON UPDATE current_timestamp() COMMENT '更新时间',
  `gmt_create` timestamp(0) NOT NULL  DEFAULT current_timestamp() COMMENT '创建时间',
  PRIMARY KEY (`id`)) ENGINE=InnoDB DEFAULT CHARSET=utf8;
DROP TABLE IF EXISTS `sys_machine`;
CREATE TABLE `sys_machine` (
  `id` varchar(32) NOT NULL  COMMENT '机器ID',
  `name` varchar(32) NOT NULL  COMMENT '名称',
  `status` Integer(3) NOT NULL  COMMENT '状态（0:离线 1:就绪 2:执行中）',
  `type` Integer(3) NOT NULL  COMMENT '类型（0:tbot）',
  `ip` varchar(32) NOT NULL  COMMENT 'ip',
  `port` varchar(32) NOT NULL  COMMENT 'port',
  `memory_size` varchar(32) COMMENT '内存大小',
  `cpu_model` varchar(32) COMMENT 'CPU型号',
  `cpu_load` decimal(10,4) COMMENT 'CPU负载',
  `mem_load` decimal(10,4) COMMENT '内存负载',
  `last_refresh_time` timestamp(0) NOT NULL  COMMENT '最新刷新时间',
  `bucket` varchar(256) NOT NULL  COMMENT 'oss存储桶',
  `gmt_update` timestamp(0) NOT NULL  DEFAULT current_timestamp() ON UPDATE current_timestamp() COMMENT '更新时间',
  `gmt_create` timestamp(0) NOT NULL  DEFAULT current_timestamp() COMMENT '创建时间',
  PRIMARY KEY (`id`)) ENGINE=InnoDB DEFAULT CHARSET=utf8;
DROP TABLE IF EXISTS `sys_running_agent`;
CREATE TABLE `sys_running_agent` (
  `id` varchar(32) NOT NULL  COMMENT '运行智能体ID',
  `name` varchar(256) NOT NULL  COMMENT '名称',
  `agent_id` varchar(32) NOT NULL  COMMENT '智能体ID',
  `running_agent_group_id` varchar(32) NOT NULL  COMMENT '运行智能体组ID',
  `organize_id` varchar(32) NOT NULL  COMMENT '组织ID',
  `status` Integer(3) NOT NULL  COMMENT '状态（0:离线 1:就绪)',
  `gmt_update` timestamp(0) NOT NULL  DEFAULT current_timestamp() ON UPDATE current_timestamp() COMMENT '更新时间',
  `gmt_create` timestamp(0) NOT NULL  DEFAULT current_timestamp() COMMENT '创建时间',
  PRIMARY KEY (`id`)) ENGINE=InnoDB DEFAULT CHARSET=utf8;
DROP TABLE IF EXISTS `sys_running_agent_group`;
CREATE TABLE `sys_running_agent_group` (
  `id` varchar(32) NOT NULL  COMMENT '运行智能体组ID',
  `name` varchar(256) NOT NULL  COMMENT '名称',
  `agent_group_id` varchar(32) NOT NULL  COMMENT '智能体组ID',
  `organize_id` varchar(32) NOT NULL  COMMENT '组织ID',
  `status` Integer(3) NOT NULL  COMMENT '状态（0:离线 1:就绪)',
  `replicas` Integer(5) NOT NULL  DEFAULT 1 COMMENT '副本数',
  `registration_center_id` varchar(32) NOT NULL  COMMENT '注册中心ID',
  `type` varchar(32) NOT NULL  DEFAULT 'agent' COMMENT '类型(agent, group, tbot)',
  `gmt_update` timestamp(0) NOT NULL  DEFAULT current_timestamp() ON UPDATE current_timestamp() COMMENT '更新时间',
  `gmt_create` timestamp(0) NOT NULL  DEFAULT current_timestamp() COMMENT '创建时间',
  PRIMARY KEY (`id`)) ENGINE=InnoDB DEFAULT CHARSET=utf8;
DROP TABLE IF EXISTS `sys_registration_center`;
CREATE TABLE `sys_registration_center` (
  `id` varchar(32) NOT NULL  COMMENT '注册中心ID',
  `name` varchar(32) NOT NULL  COMMENT '名称',
  `organize_id` varchar(32) NOT NULL  COMMENT '组织ID',
  `status` Integer(3) NOT NULL  COMMENT '状态（0:离线 1:就绪)',
  `ip` varchar(32) NOT NULL  COMMENT 'ip',
  `port` varchar(32) NOT NULL  COMMENT 'port',
  `gmt_update` timestamp(0) NOT NULL  DEFAULT current_timestamp() ON UPDATE current_timestamp() COMMENT '更新时间',
  `gmt_create` timestamp(0) NOT NULL  DEFAULT current_timestamp() COMMENT '创建时间',
  PRIMARY KEY (`id`)) ENGINE=InnoDB DEFAULT CHARSET=utf8;
DROP TABLE IF EXISTS `sys_tbot_subtask`;
CREATE TABLE `sys_tbot_subtask` (
  `id` varchar(32) NOT NULL  COMMENT 'tbot子任务ID',
  `task_id` varchar(32) NOT NULL  COMMENT '任务ID',
  `machine_id` varchar(32) COMMENT '机器ID',
  `status` Integer(3) NOT NULL  COMMENT '状态',
  `name` varchar(1024) NOT NULL  COMMENT '名称',
  `start_time` timestamp(0) NOT NULL  COMMENT '开始时间',
  `end_time` timestamp(0) NULL  COMMENT '结束时间',
  `flow_id` varchar(2048) NOT NULL  COMMENT '流程ID',
  `json_path` varchar(2048) COMMENT '模型文件',
  `gmt_update` timestamp(0) NOT NULL  DEFAULT current_timestamp() ON UPDATE current_timestamp() COMMENT '更新时间',
  `gmt_create` timestamp(0) NOT NULL  DEFAULT current_timestamp() COMMENT '创建时间',
  PRIMARY KEY (`id`)) ENGINE=InnoDB DEFAULT CHARSET=utf8;
DROP TABLE IF EXISTS `sys_agent_subtask`;
CREATE TABLE `sys_agent_subtask` (
  `id` varchar(32) NOT NULL  COMMENT '智能体子任务ID',
  `task_id` varchar(32) NOT NULL  COMMENT '任务ID',
  `name` varchar(1024) NOT NULL  COMMENT '名称',
  `status` Integer(3) NOT NULL  COMMENT '状态（0:新建 1:运行中 2:成功 3:失败 4:取消 5: 暂停 6: 人为协助）',
  `start_time` timestamp(0) NOT NULL  COMMENT '开始时间',
  `type` varchar(32) NOT NULL  COMMENT '类型(agent, group)',
  `gmt_update` timestamp(0) NOT NULL  DEFAULT current_timestamp() ON UPDATE current_timestamp() COMMENT '更新时间',
  `gmt_create` timestamp(0) NOT NULL  DEFAULT current_timestamp() COMMENT '创建时间',
  PRIMARY KEY (`id`)) ENGINE=InnoDB DEFAULT CHARSET=utf8;
