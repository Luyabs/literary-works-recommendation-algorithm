/*
 Navicat Premium Data Transfer

 Source Server         : localhost_3306
 Source Server Type    : MySQL
 Source Server Version : 80029
 Source Host           : localhost:3306
 Source Schema         : literary_works_recommendation_platform

 Target Server Type    : MySQL
 Target Server Version : 80029
 File Encoding         : 65001

 Date: 14/05/2024 12:49:42
*/

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
-- Table structure for collection
-- ----------------------------
DROP TABLE IF EXISTS `collection`;
CREATE TABLE `collection`  (
  `collection_id` bigint NOT NULL AUTO_INCREMENT COMMENT '收藏夹id',
  `owner_id` bigint NULL DEFAULT NULL COMMENT '所属用户id',
  `collection_name` varchar(40) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '收藏夹名',
  `introduction` varchar(300) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '收藏夹简介',
  `is_public` tinyint(1) NULL DEFAULT 1 COMMENT '是否公开',
  `is_default_collection` tinyint(1) NOT NULL COMMENT '是否为默认收藏夹(仅一个收藏夹可作为默认收藏夹)',
  `create_time` datetime NULL DEFAULT NULL COMMENT '创建时间',
  `update_time` datetime NULL DEFAULT NULL COMMENT '更新时间',
  PRIMARY KEY (`collection_id`) USING BTREE,
  UNIQUE INDEX `collection_name`(`collection_name`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 57 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Table structure for history_user_work
-- ----------------------------
DROP TABLE IF EXISTS `history_user_work`;
CREATE TABLE `history_user_work`  (
  `history_id` bigint NOT NULL AUTO_INCREMENT COMMENT '访问记录id',
  `user_id` bigint NULL DEFAULT NULL COMMENT '用户id',
  `work_id` bigint NULL DEFAULT NULL COMMENT '作品id',
  `visit_count` int NULL DEFAULT 1 COMMENT '访问次数',
  `create_time` datetime NULL DEFAULT NULL COMMENT '创建时间',
  `update_time` datetime NULL DEFAULT NULL COMMENT '更新时间',
  PRIMARY KEY (`history_id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 460 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Table structure for record_collection_work
-- ----------------------------
DROP TABLE IF EXISTS `record_collection_work`;
CREATE TABLE `record_collection_work`  (
  `record_id` bigint NOT NULL AUTO_INCREMENT COMMENT '收藏记录id',
  `collection_id` bigint NULL DEFAULT NULL COMMENT '收藏夹id',
  `work_id` bigint NULL DEFAULT NULL COMMENT '作品id',
  `create_time` datetime NULL DEFAULT NULL COMMENT '创建时间',
  `update_time` datetime NULL DEFAULT NULL COMMENT '更新时间',
  PRIMARY KEY (`record_id`) USING BTREE,
  INDEX `record_collection_work_collection_id_index`(`collection_id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 103 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Table structure for record_tag_work
-- ----------------------------
DROP TABLE IF EXISTS `record_tag_work`;
CREATE TABLE `record_tag_work`  (
  `record_id` bigint NOT NULL AUTO_INCREMENT COMMENT '标签关系记录id\n',
  `tag_id` bigint NULL DEFAULT NULL COMMENT '标签id\n',
  `work_id` bigint NULL DEFAULT NULL COMMENT '作品id\n',
  `create_time` datetime NULL DEFAULT NULL COMMENT '创建时间\n',
  `update_time` datetime NULL DEFAULT NULL COMMENT '更新时间\n',
  PRIMARY KEY (`record_id`) USING BTREE,
  INDEX `table_name_tag_id_index`(`tag_id`) USING BTREE,
  INDEX `table_name_work_id_index`(`work_id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 549112 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Table structure for review_user_work
-- ----------------------------
DROP TABLE IF EXISTS `review_user_work`;
CREATE TABLE `review_user_work`  (
  `review_id` bigint NOT NULL AUTO_INCREMENT COMMENT '评论id',
  `user_id` bigint NULL DEFAULT NULL COMMENT '用户id',
  `work_id` bigint NULL DEFAULT NULL COMMENT '作品id',
  `rating` int NOT NULL COMMENT '评分',
  `content` varchar(500) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '评论内容',
  `create_time` datetime NULL DEFAULT NULL COMMENT '创建时间',
  `update_time` datetime NULL DEFAULT NULL COMMENT '更新时间',
  PRIMARY KEY (`review_id`) USING BTREE,
  INDEX `review_user_work_work_id_index`(`work_id`) USING BTREE,
  INDEX `review_user_work_user_id_index`(`user_id`) USING BTREE,
  INDEX `review_user_work_rating_index`(`rating`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 7699143 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Table structure for tag
-- ----------------------------
DROP TABLE IF EXISTS `tag`;
CREATE TABLE `tag`  (
  `tag_id` bigint NOT NULL AUTO_INCREMENT COMMENT '标签id\n',
  `tag_name` varchar(300) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '标签名\n',
  `create_time` datetime NULL DEFAULT NULL COMMENT '创建时间',
  `update_time` datetime NULL DEFAULT NULL COMMENT '更新时间',
  PRIMARY KEY (`tag_id`) USING BTREE,
  UNIQUE INDEX `tag_tag_name_uindex`(`tag_name`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 87758 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Table structure for user
-- ----------------------------
DROP TABLE IF EXISTS `user`;
CREATE TABLE `user`  (
  `user_id` bigint NOT NULL AUTO_INCREMENT COMMENT '用户id',
  `username` varchar(80) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '用户名',
  `password` varchar(80) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '密码',
  `role` int NULL DEFAULT 0 COMMENT '角色{0=普通用户，1=管理员}',
  `is_banned` tinyint(1) NULL DEFAULT 0 COMMENT '账户是否被封禁',
  `is_info_public` tinyint(1) NULL DEFAULT 1 COMMENT '个人信息是否公开',
  `is_comment_public` tinyint(1) NULL DEFAULT 1 COMMENT '个人评价是否公开',
  `introduction` varchar(300) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '个人简介',
  `create_time` datetime NULL DEFAULT NULL COMMENT '创建时间',
  `update_time` datetime NULL DEFAULT NULL COMMENT '更新时间',
  PRIMARY KEY (`user_id`) USING BTREE,
  UNIQUE INDEX `username`(`username`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 738173 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Table structure for work
-- ----------------------------
DROP TABLE IF EXISTS `work`;
CREATE TABLE `work`  (
  `work_id` bigint NOT NULL AUTO_INCREMENT COMMENT '作品id',
  `tags` varchar(300) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '标签',
  `work_name` varchar(150) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '作品名',
  `author` varchar(80) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '作者',
  `introduction` varchar(300) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '作品简介',
  `publisher` varchar(80) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '出版社',
  `sum_rating` int NULL DEFAULT 0 COMMENT '总评分',
  `sum_rating_user_number` int NULL DEFAULT 0 COMMENT '总评分用户数\n',
  `cover_link` varchar(300) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '封面地址(URL)',
  `create_time` datetime NULL DEFAULT NULL COMMENT '创建时间',
  `update_time` datetime NULL DEFAULT NULL COMMENT '更新时间',
  `is_deleted` tinyint(1) NULL DEFAULT 0 COMMENT '是否被逻辑删除',
  PRIMARY KEY (`work_id`) USING BTREE,
  INDEX `work_tags_index`(`tags`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 27110871 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = Dynamic;

SET FOREIGN_KEY_CHECKS = 1;
