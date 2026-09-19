# vLLM OpenAI-compatible server, with an SSH-capable entrypoint.
#
# Upstream `vllm/vllm-openai` declares ENTRYPOINT ["vllm","serve"], and RunPod's
# `--docker-args` only replaces the CMD, so a validation bootstrap handed to that
# image would be parsed as vLLM CLI arguments. This image adds the same shim
# contract as llamacpp-prism: a single argument is run as a shell string, several
# arguments are exec'd as a server argv.
#
# Upstream's ENTRYPOINT is discarded deliberately. The shim execs the argv it is
# given, so a registry recipe still launches vLLM with its own explicit argv and the
# server keeps PID 1.
#
# openssh-server is baked in so the campaign's access-only bootstrap does not have
# to install it. The pinned digest is vllm/vllm-openai v0.29.0, linux/amd64,
# CUDA >= 13.0, inspected 2026-09-19.
FROM vllm/vllm-openai@sha256:082ca6f035279109041ffd3fe0695cb568b29bc580b35c4f297a66a08b216c1b

RUN apt-get update \
 && apt-get install -y --no-install-recommends openssh-server \
 && rm -rf /var/lib/apt/lists/* \
 && mkdir -p /run/sshd

COPY entrypoint.sh /opt/entrypoint.sh
RUN chmod 0755 /opt/entrypoint.sh

EXPOSE 8000
EXPOSE 22

ENTRYPOINT ["/opt/entrypoint.sh"]
