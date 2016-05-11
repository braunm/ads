#ifndef MB_BASE_PLUGIN
#define MB_BASE_PLUGIN <optionals.h>
#endif

#ifndef __ADS_MODULE
#define __ADS_MODULE

#include <ads_class.h>

/* Not exposing functions related to sparse Hessian, because the Hessian is dense */

RCPP_MODULE(ads){  
  Rcpp::class_< MB_Base<ads> >("ads")
    
    .constructor<const List>()
    
    .method( "get.f", & MB_Base<ads>::get_f)
    .method( "get.df", & MB_Base<ads>::get_df)
    .method( "get.fdf", & MB_Base<ads>::get_fdf)
    //    .method( "get.fdfh", & MB_Base<ads>::get_fdfh)
    .method( "record.tape", & MB_Base<ads>::record_tape)
    .method( "get.tape.stats", & MB_Base<ads>::get_tape_stats)
    .method( "get.hessian", & MB_Base<ads>::get_hessian)
    //  .method( "get.hessian.sparse", & MB_Base<ads>::get_hessian_sparse)
    //  .method( "init.hessian", & MB_Base<ads>::hessian_init_nopattern)
    //  .method( "init.sparse.hessian", & MB_Base<ads>::hessian_init_nopattern)
    //  .method( "init.hessian.pattern", & MB_Base<ads>::hessian_init)
    .method( "get.f.direct", & MB_Base<ads>::get_f_direct)
    .method( "get.LL", & MB_Base<ads>::get_LL)
    .method( "get.hyperprior", & MB_Base<ads>::get_hyperprior)
    .method( "get.hessian.test", & MB_Base<ads>::get_hessian_test)
    .method( "par.check", & MB_Base<ads>::par_check_)
    .method( "get.recursion", & MB_Base<ads>::get_recursion_)
    ;
}

#endif
