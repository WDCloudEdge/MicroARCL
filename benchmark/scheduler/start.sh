#!/usr/bin/env bash
set -Eeuo pipefail

: "${HOST_IP:?Set HOST_IP to the host address reachable from the container}"

export MINIO_ROOT_USER='AKIAIOSFODNN7EXAMPLE'
export MINIO_ROOT_PASSWORD='wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY'
export JAVA_HOME="$(dirname "$(dirname "$(readlink -f "$(command -v java)")")")"
if ! java -version 2>&1 | grep -q 'version "1.8.'; then
  echo 'Java 1.8 is required' >&2
  exit 1
fi

printf '%s %s\n' "${HOST_IP}" \
  'minio.agent.network.com db.agent.network.com taskscheduling.agent.network.com center.agent.network.com' \
  >> /etc/hosts

mkdir -p /var/run/mysqld /var/lib/mysql /data/minio
chown mysql:mysql /var/run/mysqld /var/lib/mysql

if [[ ! -d /var/lib/mysql/mysql ]]; then
  echo 'Initializing MySQL 5.7'
  mysqld --initialize-insecure --user=mysql --datadir=/var/lib/mysql
fi

pids=()
stop_all() {
  trap - TERM INT
  bash /opt/nacos/bin/shutdown.sh >/dev/null 2>&1 || true
  for pid in "${pids[@]}"; do kill "${pid}" 2>/dev/null || true; done
  wait 2>/dev/null || true
}
trap stop_all TERM INT EXIT

mysqld --user=mysql --console --bind-address=0.0.0.0 &
pids+=("$!")
for _ in $(seq 1 90); do
  if mysqladmin --protocol=socket -uroot ping >/dev/null 2>&1; then break; fi
  kill -0 "${pids[0]}" 2>/dev/null || { echo 'MySQL exited during startup' >&2; exit 1; }
  sleep 1
done
mysqladmin --protocol=socket -uroot ping >/dev/null

if [[ ! -f /var/lib/mysql/.taskScheduling-initialized ]]; then
  mysql --protocol=socket -uroot <<'SQL'
ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY '';
CREATE USER IF NOT EXISTS 'root'@'%' IDENTIFIED BY '';
GRANT ALL PRIVILEGES ON *.* TO 'root'@'%' WITH GRANT OPTION;
CREATE DATABASE IF NOT EXISTS taskScheduling CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
SQL
  mysql --protocol=socket -uroot taskScheduling < /opt/scheduler/taskScheduling.sql
  touch /var/lib/mysql/.taskScheduling-initialized
fi

echo 'Starting Nacos in standalone mode'
bash /opt/nacos/bin/startup.sh -m standalone
for _ in $(seq 1 120); do
  if curl -fsS http://127.0.0.1:8848/nacos/ >/dev/null 2>&1; then break; fi
  sleep 1
done
curl -fsS http://127.0.0.1:8848/nacos/ >/dev/null

minio server /data/minio --address ':9000' --console-address ':9001' &
pids+=("$!")
for _ in $(seq 1 60); do
  if curl -fsS http://127.0.0.1:9000/minio/health/live >/dev/null 2>&1; then break; fi
  kill -0 "${pids[1]}" 2>/dev/null || { echo 'MinIO exited during startup' >&2; exit 1; }
  sleep 1
done
curl -fsS http://127.0.0.1:9000/minio/health/live >/dev/null

java ${JAVA_OPTS:-} -jar /opt/scheduler/center.jar --spring.profiles.active=exp &
pids+=("$!")

java ${JAVA_OPTS:-} -jar /opt/scheduler/storage.jar --spring.profiles.active=exp &
pids+=("$!")
java ${JAVA_OPTS:-} -jar /opt/scheduler/taskScheduling.jar --spring.profiles.active=exp &
pids+=("$!")
tail -n +1 -F /opt/nacos/logs/start.out &
pids+=("$!")

echo 'All services started'
while true; do
  for pid in "${pids[@]}"; do
    kill -0 "${pid}" 2>/dev/null || { echo "A service exited: PID ${pid}" >&2; exit 1; }
  done
  pgrep -f '/opt/nacos/target/nacos-server.jar' >/dev/null || { echo 'Nacos exited' >&2; exit 1; }
  sleep 2
done
