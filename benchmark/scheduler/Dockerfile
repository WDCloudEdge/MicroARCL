FROM --platform=$BUILDPLATFORM golang:1.24.8 AS minio-builder
ARG TARGETARCH
RUN --mount=type=cache,target=/go/pkg/mod \
    --mount=type=cache,target=/root/.cache/go-build \
    mkdir /out \
    && CGO_ENABLED=0 GOOS=linux GOARCH="${TARGETARCH}" \
       go install github.com/minio/minio@latest \
    && minio_binary="$(find /go/bin -type f -name minio -print -quit)" \
    && test -n "${minio_binary}" \
    && cp "${minio_binary}" /out/minio

FROM ubuntu:18.04

ENV DEBIAN_FRONTEND=noninteractive TZ=Asia/Shanghai

# Ubuntu 18.04 provides MySQL 5.7 and Java 8.
RUN printf '#!/bin/sh\nexit 101\n' > /usr/sbin/policy-rc.d \
    && chmod +x /usr/sbin/policy-rc.d \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
       ca-certificates curl openjdk-8-jre-headless mysql-server-5.7 procps \
    && java -version 2>&1 | grep -q 'version "1.8.' \
    && rm -rf /var/lib/apt/lists/* /var/lib/mysql/* \
    && mkdir -p /var/lib/mysql /var/run/mysqld /data/minio /opt/scheduler \
    && chown -R mysql:mysql /var/lib/mysql /var/run/mysqld

COPY --from=minio-builder /out/minio /usr/local/bin/minio

COPY nacos-server-2.4.3.tar.gz /tmp/nacos.tar.gz
RUN tar -xzf /tmp/nacos.tar.gz -C /opt \
    && rm /tmp/nacos.tar.gz \
    && mkdir -p /opt/nacos/logs

COPY taskScheduling.sql /opt/scheduler/taskScheduling.sql
COPY center-starter-1.0.0-executable.jar /opt/scheduler/center.jar
COPY storage-starter-1.0.0-executable.jar /opt/scheduler/storage.jar
COPY taskScheduling-starter-1.0.0-executable.jar /opt/scheduler/taskScheduling.jar
COPY start.sh /usr/local/bin/start-scheduler
RUN chmod +x /usr/local/bin/start-scheduler

VOLUME ["/var/lib/mysql", "/data/minio"]
EXPOSE 3306 8848 9848 9000 9001 32196 12104 35696 38001
ENTRYPOINT ["/usr/local/bin/start-scheduler"]
