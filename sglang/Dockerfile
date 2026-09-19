# SGLang server, with an SSH-capable entrypoint.
#
# Same shim contract as llamacpp-prism and vllm-openai: one argument is a shell
# string (the campaign's validation bootstrap), several arguments are an exec'd
# server argv (what a registry recipe passes). Upstream's own ENTRYPOINT is
# replaced so RunPod's CMD-only `--docker-args` cannot turn a bootstrap into
# SGLang CLI arguments.
#
# openssh-server is baked in for the campaign's access bootstrap.
#
# Pinned digest is lmsysorg/sglang v0.5.19, linux/amd64, inspected 2026-09-19.
FROM lmsysorg/sglang@sha256:37bbbd3444732a464bbc68dee4fb0164e0ce9e18e2f027f3fc967f1152d3c262

RUN apt-get update \
 && apt-get install -y --no-install-recommends openssh-server \
 && rm -rf /var/lib/apt/lists/* \
 && mkdir -p /run/sshd

COPY entrypoint.sh /opt/entrypoint.sh
RUN chmod 0755 /opt/entrypoint.sh

EXPOSE 30000
EXPOSE 22

ENTRYPOINT ["/opt/entrypoint.sh"]
