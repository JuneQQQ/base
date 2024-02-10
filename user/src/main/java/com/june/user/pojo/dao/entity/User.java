package com.june.user.pojo.dao.entity;

import com.baomidou.mybatisplus.annotation.FieldFill;
import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;

import java.io.Serial;
import java.io.Serializable;
import java.time.LocalDateTime;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Getter;
import lombok.Setter;
import lombok.ToString;

/**
 * <p>
 * 会员
 * </p>
 *
 * @author June
 * @since 2024-02-04
 */
@Getter
@Setter
@TableName("user")
@ToString
@Schema(name = "User", description = "$!{table.comment}")
public class User implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    @Schema(description = "id")
    @TableId("id")
    private Long id;

    @TableField("nickname")
    private String nickname;

    @TableField("avatar")
    private String avatar;


    @Schema(description = "手机号")
    @TableField("phone")
    private String phone;

    @Schema(description = "密码")
    @TableField("password")
    private String password;

    @TableField(value = "create_time", fill = FieldFill.INSERT)
    private LocalDateTime createTime;

    @TableField(value = "update_time", fill = FieldFill.INSERT_UPDATE)
    private LocalDateTime updateTime;
}
