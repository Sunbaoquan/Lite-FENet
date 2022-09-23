import torch
from torch import nn
import torch.nn.functional as F
import model.resnet as resnets
import model.vgg as vgg_models
from util.util import get_vgg16_layer


class Spatial_Interaction_Module(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, expansion_factor=2):
        super(Spatial_Interaction_Module, self).__init__()
        self.stride = stride
        mid_channel = in_channel // expansion_factor
        self.identity = stride == 1 and in_channel == out_channel
        self.bottleneck = nn.Sequential(
            self.conv3x3BNRelu(in_channel, in_channel, stride=1, groups=in_channel),
            self.conv1x1BN(in_channel, mid_channel),
            self.conv1x1BNRelu(mid_channel, out_channel),
            self.conv3x3BN(out_channel, out_channel, stride=1, groups=out_channel)
        )

    def conv3x3BNRelu(self, in_channel, out_channel, stride=1, groups=1):
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, stride=stride, groups=groups, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def conv1x1BN(self, in_channel, out_channel):
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel)
        )

    def conv1x1BNRelu(self, in_channel, out_channel):
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def conv3x3BN(self, in_channel, out_channel, stride, groups):
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, stride=stride, groups=groups, bias=False),
            nn.BatchNorm2d(out_channel)
        )

    def forward(self, x):
        residual = x
        x = self.bottleneck(x)
        if self.identity:
            return x + residual
        else:
            return x


def Weighted_GAP(support_feature, mask):
    support_feature = support_feature * mask
    feat_h, feat_w = support_feature.shape[-2], support_feature.shape[-1]
    area = F.avg_pool2d(mask, (feat_h, feat_w)) * feat_h * feat_w + 0.0005
    support_feature = F.avg_pool2d(support_feature, (feat_h, feat_w)) * feat_h * feat_w / area
    return support_feature


class Lite_FENet(nn.Module):
    def __init__(self, layers=50, classes=2, zoom_factor=8,
                 criterion=nn.CrossEntropyLoss(ignore_index=255), BatchNorm=nn.BatchNorm2d,
                 backbone_pretrained=True, shot=1, scales=[60, 30, 15, 8], vgg=False):
        super(Lite_FENet, self).__init__()

        assert layers in [50, 101, 152]
        assert classes > 1

        self.zoom_factor = zoom_factor
        self.criterion = criterion
        self.shot = shot
        self.vgg = vgg
        self.scales = scales
        reduce_dim = 256
        self.num_stage = len(self.scales)
        print("The number of stage of Lite-FENet: {}".format(self.num_stage))

        if self.vgg:
            print('Using VGG16_bn')
            vgg_models.BatchNorm = BatchNorm
            vgg16 = vgg_models.vgg16_bn(pretrained=backbone_pretrained)
            self.layer0, self.layer1, self.layer2, self.layer3, self.layer4 = get_vgg16_layer(vgg16)
        else:
            print('Using ResNet - {}'.format(layers))
            resnets.BatchNorm = BatchNorm
            if layers == 50:
                resnet = resnets.resnet50(pretrained=backbone_pretrained)
            elif layers == 101:
                resnet = resnets.resnet101(pretrained=backbone_pretrained)
            else:
                resnet = resnets.resnet152(pretrained=backbone_pretrained)

            self.layer0 = nn.Sequential(
                resnet.conv1, resnet.bn1, resnet.relu1,
                resnet.conv2, resnet.bn2, resnet.relu2,
                resnet.conv3, resnet.bn3, resnet.relu3,
                resnet.maxpool)
            self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

            for n, m in self.layer3.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)

        if self.vgg:
            fea_dim = 512 + 256
        else:
            fea_dim = 1024 + 512

        self.down_query = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )

        self.down_supp = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )

        # stage1
        self.stage1 = nn.Sequential(
            nn.Conv2d(reduce_dim * 2 + 1, reduce_dim, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU(inplace=True)
        )
        # stage2
        self.transition2 = self._make_transition_layer(self.scales[:1], self.scales[:2])
        self.stage2 = self._make_stage(2, reduce_dim, reduce_dim, stride=1, expansion_factor=2)

        # stage3
        self.transition3 = self._make_transition_layer(self.scales[:2], self.scales[:3])
        self.stage3 = self._make_stage(3, reduce_dim, reduce_dim, stride=1, expansion_factor=2)

        # stage4
        self.transition4 = self._make_transition_layer(self.scales[:3], self.scales[:4])
        self.stage4 = self._make_stage(4, reduce_dim, reduce_dim, stride=1, expansion_factor=2)

        self.inner_cls = nn.ModuleList()
        for _ in range(self.num_stage):
            self.inner_cls.append(
                nn.Sequential(
                    nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(p=0.1),
                    nn.Conv2d(reduce_dim, classes, kernel_size=1)
                )
            )

        self.res1 = nn.Sequential(
            nn.Conv2d(reduce_dim * self.num_stage, reduce_dim, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU(inplace=True)
        )
        self.res2 = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.cls = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(reduce_dim, classes, kernel_size=(1, 1))
        )

    def _make_transition_layer(self, pre_stage_scales, cur_stage_scales):
        pre_scales_num = len(pre_stage_scales)
        cur_scales_num = len(cur_stage_scales)
        assert pre_scales_num < cur_scales_num
        transition_list = []

        for i in range(cur_scales_num):
            if i < pre_scales_num:
                transition_list.append(None)
            else:
                transition_list.append(
                    nn.AdaptiveAvgPool2d(cur_stage_scales[i])
                )
        return nn.ModuleList(transition_list)

    def _make_stage(self, stage, in_channel, out_channel, stride=1, expansion_factor=2):
        """parallel feature interactions"""
        modules = []
        merge_neighbor = []
        SIMs = []
        num_scales = stage
        for _ in range(num_scales - 1):
            merge_neighbor.append(
                nn.Sequential(
                    nn.Conv2d(256 * 2, 256, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False),
                    nn.ReLU(inplace=True)
                )
            )
        modules.append(nn.ModuleList(merge_neighbor))
        for _ in range(num_scales):
            SIMs.append(
                Spatial_Interaction_Module(in_channel, out_channel, stride=stride, expansion_factor=expansion_factor)
            )
        modules.append(nn.ModuleList(SIMs))
        return nn.ModuleList(modules)

    def forward(self, s_x=torch.FloatTensor(1, 1, 3, 473, 473).cuda(), x=torch.FloatTensor(1, 3, 473, 473).cuda(),
                s_y=torch.FloatTensor(1, 1, 473, 473).cuda(), y=torch.FloatTensor(1, 473, 473).cuda()):
        """
        Args:
            x: [B, 3, h, w]
            y: [B, h, w]
            s_x: [B, shot, 3, h, w]
            s_y: [B, shot, h, w]
        """
        x_size = x.size()
        assert (x_size[2] - 1) % 8 == 0 and (x_size[3] - 1) % 8 == 0
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        #   Query Feature
        with torch.no_grad():
            query_feat_0 = self.layer0(x)
            query_feat_1 = self.layer1(query_feat_0)
            query_feat_2 = self.layer2(query_feat_1)
            query_feat_3 = self.layer3(query_feat_2)
            query_feat_4 = self.layer4(query_feat_3)
            if self.vgg:
                query_feat_2 = F.interpolate(query_feat_2, size=(query_feat_3.size(2), query_feat_3.size(3)), mode='bilinear', align_corners=True)
        query_feat = torch.cat([query_feat_3, query_feat_2], 1)
        query_feat = self.down_query(query_feat)
        resize_h, resize_w = query_feat.size(2), query_feat.size(3)

        #   Support Feature
        supp_feat_list = []
        final_supp_list = []
        mask_list = []
        for i in range(self.shot):
            mask = (s_y[:, i, :, :] == 1).float().unsqueeze(1)  # [B, 1, h, w]
            mask_list.append(mask)
            with torch.no_grad():
                supp_feat_0 = self.layer0(s_x[:, i, :, :, :])
                supp_feat_1 = self.layer1(supp_feat_0)
                supp_feat_2 = self.layer2(supp_feat_1)
                supp_feat_3 = self.layer3(supp_feat_2)

                mask = F.interpolate(mask, size=(supp_feat_3.size(2), supp_feat_3.size(3)), mode='bilinear', align_corners=True)
                supp_feat_4 = self.layer4(supp_feat_3 * mask)
                final_supp_list.append(supp_feat_4)
                if self.vgg:
                    supp_feat_2 = F.interpolate(supp_feat_2, size=(supp_feat_3.size(2), supp_feat_3.size(3)), mode='bilinear', align_corners=True)
            supp_feat = torch.cat([supp_feat_3, supp_feat_2], 1)
            supp_feat = self.down_supp(supp_feat)
            supp_feat = Weighted_GAP(supp_feat, mask)
            supp_feat_list.append(supp_feat)

        # prior mask
        corr_query_mask_list = []
        cosine_eps = 1e-7
        for i, tmp_supp_feat in enumerate(final_supp_list):
            resize_size = tmp_supp_feat.size(2)
            tmp_mask = F.interpolate(mask_list[i], size=(resize_size, resize_size), mode='bilinear', align_corners=True)

            tmp_supp_feat_4 = tmp_supp_feat * tmp_mask
            q = query_feat_4
            s = tmp_supp_feat_4
            bsize, ch_sz, sp_sz, _ = q.size()[:]

            tmp_query = q
            tmp_query = tmp_query.contiguous().view(bsize, ch_sz, -1)
            tmp_query_norm = torch.norm(tmp_query, 2, 1, True)

            tmp_supp = s
            tmp_supp = tmp_supp.contiguous().view(bsize, ch_sz, -1)
            tmp_supp = tmp_supp.contiguous().permute(0, 2, 1)
            tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True)

            similarity = torch.bmm(tmp_supp, tmp_query) / (torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)
            similarity = similarity.max(1)[0].view(bsize, sp_sz * sp_sz)
            similarity = (similarity - similarity.min(1)[0].unsqueeze(1)) / (
                    similarity.max(1)[0].unsqueeze(1) - similarity.min(1)[0].unsqueeze(1) + cosine_eps)
            corr_query = similarity.view(bsize, 1, sp_sz, sp_sz)
            corr_query = F.interpolate(corr_query, size=(query_feat_3.size(2), query_feat_3.size(3)), mode='bilinear', align_corners=True)
            corr_query_mask_list.append(corr_query)

        corr_query_mask = torch.cat(corr_query_mask_list, 1).mean(1).unsqueeze(1)
        corr_query_mask = F.interpolate(corr_query_mask, size=(resize_h, resize_w), mode='bilinear', align_corners=True)

        if self.shot > 1:
            supp_feat = supp_feat_list[0]
            for i in range(1, len(supp_feat_list)):
                supp_feat += supp_feat_list[i]
            supp_feat /= len(supp_feat_list)

        # ------------------stage1------------------
        # To keep same spatial size
        # Query [B, 256, 60, 60] -> [B, 256, 60, 60]
        query_feat = nn.AdaptiveAvgPool2d(self.scales[0])(query_feat)
        # Support [B, 256, 1, 1] -> [B, 256, 60, 60]
        supp_feat = supp_feat.expand(-1, -1, self.scales[0], self.scales[0])
        # Prior Mask [B, 1, 60, 60] -> [B, 1, 60, 60]
        corr_mask = F.interpolate(corr_query_mask, size=(self.scales[0], self.scales[0]), mode='bilinear', align_corners=True)
        merge_feat = torch.cat([query_feat, supp_feat, corr_mask], 1)
        merge_feat = self.stage1(merge_feat)

        # ------------------stage2------------------
        input_list = []    # store this stage input
        for i in range(2):
            if self.transition2[i] is not None:
                input_list.append(self.transition2[i](merge_feat))
            else:
                input_list.append(merge_feat)
        pyramid_feat_list = []
        output_list = []    # store this stage output refined features
        stage2_cls_out = []   # store this stage cls results
        for idx, x in enumerate(input_list):
            if idx >= 1:
                pre_refine_feat = pyramid_feat_list[idx - 1].clone()
                pre_refine_feat = F.interpolate(pre_refine_feat, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)
                cat_feat = torch.cat([x, pre_refine_feat], 1)
                x = self.stage2[0][idx - 1](cat_feat) + x
            x = self.stage2[1][idx](x)
            output_list.append(x)
            stage2_cls_out.append(self.inner_cls[idx](x))
            x = F.interpolate(x, size=(resize_h, resize_w), mode='bilinear', align_corners=True)
            pyramid_feat_list.append(x)

        # ------------------stage3------------------
        input_list = []
        for i in range(3):
            if self.transition3[i] is not None:
                input_list.append(self.transition3[i](output_list[-1]))
            else:
                input_list.append(output_list[i])
        pyramid_feat_list = []
        output_list = []
        stage3_cls_out = []
        for idx, x in enumerate(input_list):
            if idx >= 1:
                pre_refine_feat = pyramid_feat_list[idx - 1].clone()
                pre_refine_feat = F.interpolate(pre_refine_feat, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)
                cat_feat = torch.cat([x, pre_refine_feat], 1)
                x = self.stage3[0][idx - 1](cat_feat) + x
            x = self.stage3[1][idx](x)
            output_list.append(x)
            stage3_cls_out.append(self.inner_cls[idx](x))
            x = F.interpolate(x, size=(resize_h, resize_w), mode='bilinear', align_corners=True)
            pyramid_feat_list.append(x)

        # ------------------stage4------------------
        input_list = []
        for i in range(4):
            if self.transition4[i] is not None:
                input_list.append(self.transition4[i](output_list[-1]))
            else:
                input_list.append(output_list[i])
        pyramid_feat_list = []
        stage4_cls_out = []
        for idx, x in enumerate(input_list):
            if idx >= 1:
                pre_refine_feat = pyramid_feat_list[idx - 1].clone()
                pre_refine_feat = F.interpolate(pre_refine_feat, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)
                cat_feat = torch.cat([x, pre_refine_feat], 1)
                x = self.stage4[0][idx - 1](cat_feat) + x
            x = self.stage4[1][idx](x)
            stage4_cls_out.append(self.inner_cls[idx](x))
            x = F.interpolate(x, size=(resize_h, resize_w), mode='bilinear', align_corners=True)
            pyramid_feat_list.append(x)


        query_feat = torch.cat(pyramid_feat_list, 1)
        query_feat = self.res1(query_feat)
        query_feat = self.res2(query_feat) + query_feat
        out = self.cls(query_feat)

        # Output Part
        if self.zoom_factor != 1:
            out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)  # [B, 2, h, w]

        if self.training:  # model.train()
            main_loss = self.criterion(out, y.long())
            stage2_loss = torch.zeros_like(main_loss).cuda()
            stage3_loss = torch.zeros_like(main_loss).cuda()
            stage4_loss = torch.zeros_like(main_loss).cuda()

            for idx in range(len(stage2_cls_out)):
                inner_out = stage2_cls_out[idx]
                inner_out = F.interpolate(inner_out, size=(h, w), mode='bilinear', align_corners=True)  # [B, 2, h, w]
                stage2_loss = stage2_loss + self.criterion(inner_out, y.long())
            for idx in range(len(stage3_cls_out)):
                inner_out = stage3_cls_out[idx]
                inner_out = F.interpolate(inner_out, size=(h, w), mode='bilinear', align_corners=True)
                stage3_loss = stage3_loss + self.criterion(inner_out, y.long())
            for idx in range(len(stage4_cls_out)):
                inner_out = stage4_cls_out[idx]
                inner_out = F.interpolate(inner_out, size=(h, w), mode='bilinear', align_corners=True)
                stage4_loss = stage4_loss + self.criterion(inner_out, y.long())

            # aux loss mean
            aux_loss = stage2_loss / len(stage2_cls_out) + stage3_loss / len(stage3_cls_out) + stage4_loss \
                       / len(stage4_cls_out)
            return out.max(1)[1], main_loss, aux_loss
        else:
            return out
