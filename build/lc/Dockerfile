# Copyright 2021 The KubeEdge Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Add cross buildx improvement
# LC has built sqlite3 which requires CGO with CGO_ENABLED=1
# _speed_buildx_for_cgo_alpine_
FROM golang:1.22.9-alpine3.19 AS builder
LABEL stage=builder

ARG GO_LDFLAGS

# install build tools
RUN apk update
RUN apk add build-base bash

WORKDIR /code
# copy source
COPY . .
RUN make build WHAT=lc GO_LDFLAGS=$GO_LDFLAGS OUT_DIR=_output


FROM alpine:3.19

COPY --from=builder /code/_output/bin/sedna-lc /usr/local/bin/sedna-lc

CMD ["sedna-lc"]
