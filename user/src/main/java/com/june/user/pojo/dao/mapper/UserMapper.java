package com.june.user.pojo.dao.mapper;

import com.june.user.pojo.dao.entity.User;
import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import org.apache.ibatis.annotations.Mapper;

/**
 * <p>
 * 会员 Mapper 接口
 * </p>
 *
 * @author June
 * @since 2024-02-04
 */
@Mapper
public interface UserMapper extends BaseMapper<User> {

}
