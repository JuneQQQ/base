package com.june.user.service;

import com.june.user.pojo.dao.entity.User;
import com.baomidou.mybatisplus.extension.service.IService;
import com.june.user.pojo.vo.LoginFormVo;
import jakarta.servlet.http.HttpServletRequest;

/**
 * <p>
 * 会员 服务类
 * </p>
 *
 * @author June
 */
public interface IUserService extends IService<User> {

    String sendCode(String phone);

    String login(LoginFormVo vo);

    Long register(User user);

    User getUserInfo(HttpServletRequest request);

    void logout(HttpServletRequest request);
}
