DROP TABLE IF EXISTS `sys_task`;
CREATE TABLE `sys_task` (
  `id` varchar(32) NOT NULL  COMMENT 'Task ID',
  `name` varchar(4096) NOT NULL  COMMENT 'Name',
  `status` Integer(3) COMMENT 'Status (0: new, 1: running, 2: success, 3: failed, 4: canceled, 5: paused, 6: manual assistance, 7: completed, 8: completion failed)',
  `user_id` varchar(32) COMMENT 'User ID',
  `organize_id` varchar(32) COMMENT 'Organization ID',
  `graph_id` varchar(32) COMMENT 'Execution graph ID',
  `gmt_update` timestamp(0) NOT NULL  DEFAULT current_timestamp() ON UPDATE current_timestamp() COMMENT 'Updated at',
  `gmt_create` timestamp(0) NOT NULL  DEFAULT current_timestamp() COMMENT 'Created at',
  PRIMARY KEY (`id`)) ENGINE=InnoDB DEFAULT CHARSET=utf8;
DROP TABLE IF EXISTS `sys_graph`;
CREATE TABLE `sys_graph` (
  `id` varchar(32) NOT NULL  COMMENT 'Execution graph ID',
  `file_path` varchar(1024) NOT NULL  COMMENT 'File',
  `gmt_update` timestamp(0) NOT NULL  DEFAULT current_timestamp() ON UPDATE current_timestamp() COMMENT 'Updated at',
  `gmt_create` timestamp(0) NOT NULL  DEFAULT current_timestamp() COMMENT 'Created at',
  PRIMARY KEY (`id`)) ENGINE=InnoDB DEFAULT CHARSET=utf8;
DROP TABLE IF EXISTS `sys_machine`;
CREATE TABLE `sys_machine` (
  `id` varchar(32) NOT NULL  COMMENT 'Machine ID',
  `name` varchar(32) NOT NULL  COMMENT 'Name',
  `status` Integer(3) NOT NULL  COMMENT 'Status (0: offline, 1: ready, 2: running)',
  `type` Integer(3) NOT NULL  COMMENT 'Type (0: tbot)',
  `ip` varchar(32) NOT NULL  COMMENT 'ip',
  `port` varchar(32) NOT NULL  COMMENT 'port',
  `memory_size` varchar(32) COMMENT 'Memory size',
  `cpu_model` varchar(32) COMMENT 'CPU model',
  `cpu_load` decimal(10,4) COMMENT 'CPU load',
  `mem_load` decimal(10,4) COMMENT 'Memory load',
  `last_refresh_time` timestamp(0) NOT NULL  COMMENT 'Last refresh time',
  `bucket` varchar(256) NOT NULL  COMMENT 'OSS bucket',
  `gmt_update` timestamp(0) NOT NULL  DEFAULT current_timestamp() ON UPDATE current_timestamp() COMMENT 'Updated at',
  `gmt_create` timestamp(0) NOT NULL  DEFAULT current_timestamp() COMMENT 'Created at',
  PRIMARY KEY (`id`)) ENGINE=InnoDB DEFAULT CHARSET=utf8;
DROP TABLE IF EXISTS `sys_running_agent`;
CREATE TABLE `sys_running_agent` (
  `id` varchar(32) NOT NULL  COMMENT 'Running agent ID',
  `name` varchar(256) NOT NULL  COMMENT 'Name',
  `agent_id` varchar(32) NOT NULL  COMMENT 'Agent ID',
  `running_agent_group_id` varchar(32) NOT NULL  COMMENT 'Running agent group ID',
  `organize_id` varchar(32) NOT NULL  COMMENT 'Organization ID',
  `status` Integer(3) NOT NULL  COMMENT 'Status (0: offline, 1: ready)',
  `gmt_update` timestamp(0) NOT NULL  DEFAULT current_timestamp() ON UPDATE current_timestamp() COMMENT 'Updated at',
  `gmt_create` timestamp(0) NOT NULL  DEFAULT current_timestamp() COMMENT 'Created at',
  PRIMARY KEY (`id`)) ENGINE=InnoDB DEFAULT CHARSET=utf8;
DROP TABLE IF EXISTS `sys_running_agent_group`;
CREATE TABLE `sys_running_agent_group` (
  `id` varchar(32) NOT NULL  COMMENT 'Running agent group ID',
  `name` varchar(256) NOT NULL  COMMENT 'Name',
  `agent_group_id` varchar(32) NOT NULL  COMMENT 'Agent group ID',
  `organize_id` varchar(32) NOT NULL  COMMENT 'Organization ID',
  `status` Integer(3) NOT NULL  COMMENT 'Status (0: offline, 1: ready)',
  `replicas` Integer(5) NOT NULL  DEFAULT 1 COMMENT 'Replica count',
  `registration_center_id` varchar(32) NOT NULL  COMMENT 'Registry ID',
  `type` varchar(32) NOT NULL  DEFAULT 'agent' COMMENT 'Type (agent, group, tbot)',
  `gmt_update` timestamp(0) NOT NULL  DEFAULT current_timestamp() ON UPDATE current_timestamp() COMMENT 'Updated at',
  `gmt_create` timestamp(0) NOT NULL  DEFAULT current_timestamp() COMMENT 'Created at',
  PRIMARY KEY (`id`)) ENGINE=InnoDB DEFAULT CHARSET=utf8;
DROP TABLE IF EXISTS `sys_registration_center`;
CREATE TABLE `sys_registration_center` (
  `id` varchar(32) NOT NULL  COMMENT 'Registry ID',
  `name` varchar(32) NOT NULL  COMMENT 'Name',
  `organize_id` varchar(32) NOT NULL  COMMENT 'Organization ID',
  `status` Integer(3) NOT NULL  COMMENT 'Status (0: offline, 1: ready)',
  `ip` varchar(32) NOT NULL  COMMENT 'ip',
  `port` varchar(32) NOT NULL  COMMENT 'port',
  `gmt_update` timestamp(0) NOT NULL  DEFAULT current_timestamp() ON UPDATE current_timestamp() COMMENT 'Updated at',
  `gmt_create` timestamp(0) NOT NULL  DEFAULT current_timestamp() COMMENT 'Created at',
  PRIMARY KEY (`id`)) ENGINE=InnoDB DEFAULT CHARSET=utf8;
DROP TABLE IF EXISTS `sys_tbot_subtask`;
CREATE TABLE `sys_tbot_subtask` (
  `id` varchar(32) NOT NULL  COMMENT 'Tbot subtask ID',
  `task_id` varchar(32) NOT NULL  COMMENT 'Task ID',
  `machine_id` varchar(32) COMMENT 'Machine ID',
  `status` Integer(3) NOT NULL  COMMENT 'Status',
  `name` varchar(1024) NOT NULL  COMMENT 'Name',
  `start_time` timestamp(0) NOT NULL  COMMENT 'Start time',
  `end_time` timestamp(0) NULL  COMMENT 'End time',
  `flow_id` varchar(2048) NOT NULL  COMMENT 'Flow ID',
  `json_path` varchar(2048) COMMENT 'Model file',
  `gmt_update` timestamp(0) NOT NULL  DEFAULT current_timestamp() ON UPDATE current_timestamp() COMMENT 'Updated at',
  `gmt_create` timestamp(0) NOT NULL  DEFAULT current_timestamp() COMMENT 'Created at',
  PRIMARY KEY (`id`)) ENGINE=InnoDB DEFAULT CHARSET=utf8;
DROP TABLE IF EXISTS `sys_agent_subtask`;
CREATE TABLE `sys_agent_subtask` (
  `id` varchar(32) NOT NULL  COMMENT 'Agent subtask ID',
  `task_id` varchar(32) NOT NULL  COMMENT 'Task ID',
  `name` varchar(1024) NOT NULL  COMMENT 'Name',
  `status` Integer(3) NOT NULL  COMMENT 'Status (0: new, 1: running, 2: success, 3: failed, 4: canceled, 5: paused, 6: manual assistance)',
  `start_time` timestamp(0) NOT NULL  COMMENT 'Start time',
  `type` varchar(32) NOT NULL  COMMENT 'Type (agent, group)',
  `gmt_update` timestamp(0) NOT NULL  DEFAULT current_timestamp() ON UPDATE current_timestamp() COMMENT 'Updated at',
  `gmt_create` timestamp(0) NOT NULL  DEFAULT current_timestamp() COMMENT 'Created at',
  PRIMARY KEY (`id`)) ENGINE=InnoDB DEFAULT CHARSET=utf8;
