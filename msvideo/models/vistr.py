# # Copyright 2022 Huawei Technologies Co., Ltd
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# # http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# # ============================================================================
# """VisTR."""
# from typing import Optional
# import ml_collections as collections

# from mindspore import nn
# # from msvideo.models.backbones import vistr_backbone
# # from msvideo.models.embedding import vistr_embed
# from msvideo.models.head import vistr_head
# from msvideo.models.base import BaseRecognizer
# from msvideo.classification.utils.model_urls import model_urls
# from msvideo.utils.load_pretrained_model import LoadPretrainedModel

# __all__ = [
#     'vistr',
#     'vistr_r50',
#     # TODO: supplement vistr_r101
# ]


# def vistr(name: str,
#           train_embeding: bool,
#           num_queries: int,
#           num_pos_feats: int = 64,
#           num_frames: int = 64,
#           temperature: int = 10000,
#           normalize: bool = True,
#           scale: float = None,
#           hidden_dim: int = 384,
#           d_model: int = 256,
#           nhead: int = 8,
#           num_encoder_layers: int = 6,
#           num_decoder_layers: int = 6,
#           dim_feedforward: int = 2048,
#           dropout: int = 0.1,
#           activation: str = "relu",
#           normalize_before: bool = False,
#           return_intermediate_dec: bool = False,
#           aux_loss: bool = False,
#           num_class: int = 41,
#           pretrained: bool = False,
#           arch: Optional[str] = None) -> nn.Cell:
#     """
#     Vistr Architecture.
#     """
#     embedding = vistr_embed.VistrEmbeding(name=name,
#                                           train_embeding=train_embeding,
#                                           num_pos_feats=num_pos_feats,
#                                           num_frames=num_frames,
#                                           temperature=temperature,
#                                           normalize=normalize,
#                                           scale=scale,
#                                           hidden_dim=hidden_dim)
#     backbone = vistr_backbone.VistrBackbone(d_model=d_model,
#                                             nhead=nhead,
#                                             num_encoder_layers=num_encoder_layers,
#                                             num_decoder_layers=num_decoder_layers,
#                                             dim_feedforward=dim_feedforward,
#                                             dropout=dropout,
#                                             activation=activation,
#                                             normalize_before=normalize_before,
#                                             return_intermediate_dec=return_intermediate_dec,
#                                             aux_loss=aux_loss,
#                                             num_class=num_class)
#     head = vistr_head.vistrsegm(d_model=d_model,
#                                 nhead=nhead,
#                                 num_frames=num_frames,
#                                 num_queries=num_queries)
#     model = BaseRecognizer(backbone=backbone, embedding=embedding, head=head)
#     if pretrained:
#         # Download the pre-trained checkpoint file from url, and load ckpt file.
#         # TODO: model_urls is not defined yet.
#         LoadPretrainedModel(model, model_urls[arch]).run()
#     return model


# def vistr_r50(name='ResNet50',
#               train_embeding=True,
#               d_model=384,
#               return_intermediate_dec=True,
#               num_class=41,
#               aux_loss=True,
#               nhead=8,
#               num_frames=36,
#               num_queries=360,
#               pretrained=True
#               ) -> nn.Cell:
#     config = collections.ConfigDict()
#     config.arch = "vistr_r50_vos"
#     config.name = name
#     config.train_embeding = train_embeding
#     config.d_model = d_model
#     config.return_intermediate_dec = return_intermediate_dec
#     config.num_class = num_class
#     config.aux_loss = aux_loss
#     config.nhead = nhead
#     config.num_frames = num_frames
#     config.num_queries = num_queries
#     config.pretrained = pretrained
#     return vistr(**config)
