/*
 * Represent a symbol in scope
 */

package org.homeunix.wap.XSS;

import java.io.IOException;
import org.homeunix.wap.table.tainted.RootTaintedSymbol;
import org.homeunix.wap.table.tainted.UntaintedTable;
import org.homeunix.wap.table.tainted.TaintedTable;
import org.homeunix.wap.table.tainted.VariableTaintedSymbol;
import org.homeunix.wap.table.tainted.RelatedTaintedSymbol;
import org.homeunix.wap.table.symbol.ArgFunctionSymbol;
import org.homeunix.wap.table.symbol.SymbolTable;
import org.homeunix.wap.table.symbol.Scope;
import org.homeunix.wap.table.symbol.ClassSymbol;
import org.homeunix.wap.table.symbol.ClassTable;
import org.homeunix.wap.table.symbol.Instance;
import org.homeunix.wap.table.symbol.ScopedSymbol;
import org.homeunix.wap.table.symbol.MethodSymbolCall;
import org.homeunix.wap.table.symbol.MethodClass;
import org.homeunix.wap.table.symbol.InstanceTable;
import org.homeunix.wap.table.symbol.Symbol;
import org.homeunix.wap.table.symbol.MethodSymbol;
import org.homeunix.wap.table.symbol.MethodTable;
import java.util.*;


/**
 *
 * @author Iberia Medeiros
 */
//public class FuncAndIncCodeInj extends CallSymbol{
public class FuncAndIncXSS extends ScopedSymbol{    

    // *** CONSTRUCTORS
    public FuncAndIncXSS(Scope scp) {
        super(scp.getScopeSymbol().getName(), scp.getScopeSymbol().getCodeLine(), scp.getScopeSymbol().getTainted(), scp.getEnclosingScope(), scp.getScopeSymbol().getFileSymbol());   
    }
   

    // *** METHODS
    public void resolveSymbolClassXSS(MethodSymbol ms, Scope scp, TaintedTable mts, UntaintedTable mus, String filename, MethodSymbolCall mtsc, Instance inst, Map mct, Boolean ftime, Scope cs_att, Scope cs_acc, String typeMeth, Boolean isClassInstruction) throws IOException{
    if (scp.getScopeSymbol().getIsSecureFunction() == false){ 
    Scope csAtt_acc = cs_att;
    Scope csAtt_att = cs_acc;
    String s = scp.getScopeName();
    Instance ret = null;
    Boolean exit_for = false;
    
    if (inst != null){   
    SymbolTable t = inst.getInstanceAttributes();
  
    //pegar no atributo do objecto numa atribuicao ou acesso
    if (t.containsSymbol(s) || (scp.getEnclosingScope() != null && scp.getEnclosingScope().getScopeName().equals("this") && t.containsSymbol(s))){
       //pegar no atributo do objecto numa atribuicao ou acesso
       if (ftime == true){
           csAtt_att = (Scope) t.getSymbolOfListSymbols(s);
           try{
               csAtt_att.getMembers().clear();
               for(int i=0; i< scp.getMembers().size(); i++)
                    csAtt_att.getMembers().add(scp.getMembers().get(i));
           } catch (Exception e) {}
           ftime = false;

           if (scp.getEnclosingScope() != null && scp.getEnclosingScope().getScopeName().equals("this")){
                Symbol sym_this = scp.getEnclosingScope().getScopeSymbol();
                scp.getScopeSymbol().setIsInstance(sym_this.getIsInstance());
                scp.getScopeSymbol().setIsClone(sym_this.getIsClone());
           }
       }
       else{
           //csAtt_acc = (Scope) t.getMembers().get(i);
           csAtt_acc = (Scope) t.getSymbolOfListSymbols(s);
       }
        }

    }

    Iterator <Symbol> it;
    Symbol sym;
    for (it = scp.getMembers().iterator(); it.hasNext();){
        sym = it.next();
        if (isClassInstruction == true){
        int op = 0;

        // criacao de um objecto
        // do tipo $a = new A_class()
        if (scp.getScopeSymbol().getIsInstance() == true)
            op = 2;

        // atribuicao e' um clone
        // do tipo $a = clone $obj
        if (scp.getScopeSymbol().getIsClone() == true)
            op = 3;

        // staticMember
        // A::meth()
        if (scp.getScopeSymbol().getIsStaticMember() == true)
            op = 4;

        // chamada de um metodo
        if(scp.getScopeSymbol().getIsFunction() == true && scp.getScopeSymbol().getIsClassMethod() == true &&
           scp.getScopeSymbol().getIsSecureFunction()==false && scp.getScopeSymbol().getIsTaintedFunction()==false){
            if (ms != null)
                op = 5; // metodo dentro de um metodo
            else
                op = 1; //metodo normal
        }

        switch (op){
            case 1:{ // chamada de um metodo normal
                ftime = false;
                
                // 1. obter instancia
                String instName = scp.getEnclosingScope().getScopeName();
                String fileInst = scp.getEnclosingScope().getScopeSymbol().getFileSymbol();
                InstanceTable instTable = GlobalDataXSS.MainInstancesTable.get(fileInst);
                SymbolTable st = GlobalDataXSS.MainSymbolTable.get(fileInst);
                String instName_aux = instName;
                Instance inst1 = null;
                if (st.getAliases().containsKey(instName_aux) == true)
                        instName_aux = st.getInstanceNameFromAliases(instName_aux, instTable);                
                
                if (instTable != null && instTable.containsInstance(instName_aux) == true)
                    inst1 = (Instance) instTable.getInstanceOfListInstances(instName_aux);

                // 2. obter classe
                String c_name = inst1.getInstanceClass();
                String cf_name = inst1.getFileOfClass();
                ClassTable ct = GlobalDataXSS.MainClassesTable.get(cf_name);
                ClassSymbol currentClass = ct.getClassOfListClasses(c_name);

                // * 3. procurar metodo
                MethodClass mc = currentClass.getMethodClass(scp.getScopeName(), "", mct);
                MethodSymbol mss_final = mc.getMeth();
                String mss_final_file = mc.getClassMeth().getFileClass();

                ret = null;
                ret = instanceMethod(mss_final, scp, filename, mss_final_file, mts, mus, inst1, null, mct, "", ret, instTable, st);                

                if (ret != null && instTable.containsInstance(ret.getInstanceName()) == true){
                        Boolean a = true;
                        Scope enc_f = scp;
                        Scope enc_b;
                        while (a == true){
                                enc_b = enc_f.getEnclosingScope();
                                if (enc_b.getScopeName().equals(st.getScopeName()) == false)
                                        enc_f = enc_b;
                                else
                                        a = false;
                        }

                        Symbol sy = new Symbol(ret.getInstanceName(), ret.getTainted(), ret.getCreateLineInstance(), ret.getFileInstance());
                        sy.setIsClassInstruction(true);
                        st.getAliases().put(enc_f.getScopeName(), sy);
                        scp.resolveSymbol(scp, sym, mts, mus, sym.getFileSymbol());
                        mts.removeDuplicates(sym);

                        if (cs_att != null){
                                cs_att.getScopeSymbol().setTainted(ret.getTainted());
                                inst1.updateTainted(cs_att, sym);
                        }
                }
                exit_for = true;
                //break;
                return;
            }
            case 2:{
                ftime = false;
                // criacao de um objecto

                Instance inst1 = null;
                InstanceTable instances_table = null;
                if (ms == null){
                    String file_inst = sym.getFileSymbol();
                    instances_table = GlobalDataXSS.MainInstancesTable.get(file_inst);
                    Iterator <Instance> it1 = instances_table.getMembers().iterator();
                    for (; it1.hasNext();){
                        inst1 = it1.next();
                        if (inst1.getInstanceName().equals(s) && inst1.getCreateLineInstance() == sym.getCodeLine()){
                            break;
                        }
                    }
                }

                /* 1. Procurar classe */
                ClassTable class_table = null;
                ClassSymbol currentClass = null;

                for (Iterator <ClassTable> it3 = GlobalDataXSS.MainClassesTable.values().iterator(); it3.hasNext();){
                    class_table = it3.next();

                    if (class_table.containsClass(sym.getName()) == true){
                        currentClass = (ClassSymbol) class_table.getClassOfListClasses(sym.getName());
                        break;
                    }
                }
                
                    if (ms != null){
                        if (typeMeth.equals("constructor") == true)
                            instances_table = inst.getObjectsOfInstance();
                        else
                            instances_table = mtsc.getObjectsOfMethod();
                    }

                    if (currentClass.getExtendsClassName() != null){
                        inst1 = new Instance(scp.getScopeName(), scp.getScopeSymbol().getCodeLine(), inst.getFileInstance(), sym.getName(), class_table.getScopeName(), currentClass, true);
                        instances_table.define(inst1);
                        return;
                    }
                    if (ms != null){
                        inst1 = new Instance(scp.getScopeName(), scp.getScopeSymbol().getCodeLine(), inst.getFileInstance(), sym.getName(), class_table.getScopeName(), currentClass, false);
                        instances_table.define(inst1);
                    }

                /* 2. procurar metodo construtor*/
                MethodClass mc = currentClass.getMethodClass(sym.getName(), "construct", mct);
                MethodSymbol mss_final = mc.getMeth();
                String mss_final_file = mc.getClassMeth().getFileClass();

                /* Fazer a analise do construtor */
                if (mss_final == null){ // a classe nao tem metodo construtor. logo fazer taint analise aos atributos
                    SymbolTable t1 = inst1.getInstanceAttributes();
                    TaintedTable tt1 = inst1.getMtsInstanceAttributes();
                    UntaintedTable ut1 = inst1.getMusInstanceAttributes();
                    Scope scp1;
                    for(Iterator <Symbol> it4 = t1.getMembers().iterator(); it4.hasNext();){
                         scp1 = (Scope) it4.next();
                         scp1.resolveSymbolClass(tt1, ut1, inst1.getFileOfClass());
                    }
                }
                else{
                    typeMeth = "constructor";
                    MethodSymbolXSS meth_sym = new MethodSymbolXSS(mss_final);
                    int valueFinal = meth_sym.analyseMethodXSS(mss_final,scp.getScopeSymbol().getCodeLine(), (Scope)sym, mts, mus, filename, mss_final_file, inst1, mct, typeMeth);
                    scp.getScopeSymbol().setTainted(valueFinal);
                    scp.resolveSymbol(scp, sym, mts, mus, scp.getScopeSymbol().getFileSymbol());

                    if (cs_att != null){
                        cs_att.getScopeSymbol().setTainted(valueFinal);
                        inst.updateTainted(cs_att, sym);
                    }
                }
                return;
            }

            case 3:{
                ftime = false;

                // atribuicao e' um clone
                if (ms != null)
                    mtsc.getAliases().put(s, sym);
                
                scp.resolveSymbol(scp, sym, mts, mus, sym.getFileSymbol());
                if (cs_att != null){
                    cs_att.getScopeSymbol().setTainted(scp.getScopeSymbol().getTainted());
                    inst.updateTainted(cs_att, sym);
                }
                return;
            }

            case 4:{
                ftime = false;

                // staticMember
                // obter classe
                String clas = "";
                String clas_file = "";
                if (s.equals("self") == true){
                    clas = inst.getInstanceClass();
                    clas_file = inst.getFileOfClass();
                }
                else{
                    if (s.equals("parent") == true){
                        clas = inst.getInstanceClass();
                        clas_file = inst.getFileOfClass();
                        ClassTable ct = GlobalDataXSS.MainClassesTable.get(clas_file);
                        ClassSymbol cls = (ClassSymbol) ct.getClassOfListClasses(clas);
                        clas = cls.getExtendsClassName();
                        clas_file = cls.getFileExtendsClass();

                    }
                    else{
                        // obter nome e file da class
                        for (Iterator <ClassTable> it1 = mct.values().iterator(); it1.hasNext();){
                            ClassTable class_table = it1.next();
                            if (class_table.containsClass(s) == true){
                                clas = s;
                                clas_file = class_table.getScopeName();
                                break;
                            }
                        }
                    }
                }
                if (clas.equals("") == false && sym.getIsClassMethod() == true){
                    ClassTable class_table = (ClassTable) mct.get(clas_file);
                    ClassSymbol currentClass  = (ClassSymbol) class_table.getClassOfListClasses(clas);

                    // obter metodo
                    //Symbol sy = (Symbol) scp.getMembers().get(0);
                    String ss = "";
                    if (sym.getName().equals("__construct"))
                        ss = "construct";
                    MethodClass mc = currentClass.getMethodClass(sym.getName(), ss, mct);
                    MethodSymbol mss_final = mc.getMeth();
                    String mss_final_file = mc.getClassMeth().getFileClass();
                    /* Fazer a analise do metodo */
                    if (mss_final != null){
                        Instance inst1 = null;
                        if (s.equals("self") == true || s.equals("parent") == true)
                            inst1 = inst;
                        
                        Scope scp1 = (Scope) sym;
                        MethodSymbolXSS meth_sym = new MethodSymbolXSS(mss_final);
                        int valueFinal = meth_sym.analyseMethodXSS(mss_final,scp.getScopeSymbol().getCodeLine(), scp1, mts, mus, filename, mss_final_file, inst, mct, typeMeth);
                        scp1.getScopeSymbol().setTainted(valueFinal);
                        scp.resolveSymbol(scp, scp1.getScopeSymbol(), mts, mus, scp1.getScopeSymbol().getFileSymbol());
                    }
                }
                return;
            }

            case 5:{
                ftime = false;
                InstanceTable instTable = null;
                SymbolTable st = null;
                
                ret = instanceMethod(ms, scp, filename, "", mts, mus, inst, mtsc, mct, typeMeth, ret, null, null);

                InstanceTable instances_table = null;
                if (typeMeth.equals("constructor") == true)
                    instances_table = inst.getObjectsOfInstance();
                else
                    instances_table = mtsc.getObjectsOfMethod();

                //if (ret != null && instances_table.getMembers().contains(ret) == true){
                if (ret != null && instances_table.containsInstance(ret.getInstanceName()) == true){
                    Boolean a = true;
                    Scope enc_f = scp;
                    Scope enc_b;
                    while (a == true){
                        enc_b = enc_f.getEnclosingScope();
                        if (enc_b.getScopeName().equals(ms.getFunctionName()) == false)
                            enc_f = enc_b;
                        else
                            a = false;
                    }

                    Symbol sy = new Symbol(ret.getInstanceName(), ret.getTainted(), ret.getCreateLineInstance(), ret.getFileInstance());
                    sy.setIsClassInstruction(true);
                    mtsc.getAliases().put(enc_f.getScopeName(), sy);
                    scp.resolveSymbol(scp, sym, mts, mus, sym.getFileSymbol());
                    mts.removeDuplicates(sym);

                    if (cs_att != null){
                        cs_att.getScopeSymbol().setTainted(ret.getTainted());
                        inst.updateTainted(cs_att, sym);
                    }
                }
                exit_for = true;
                return;
            }
            default:{
                if (scp.getEnclosingScope() != null && scp.getEnclosingScope().getScopeName().equals("this") == false)
                    ftime = false;

                if (sym.getRootScope() == null && (sym.getIsClassMethod() == false && sym.getIsUserFunction() == false)){
                    if (ms != null){                  
                      if (ms.getArgsFunction().containsKey(sym.getName()) == true){
                          ArgFunctionSymbol arg = (ArgFunctionSymbol) ms.getArgsFunction().get(sym.getName());
                          sym.setName(arg.getCallFunctionArg());
                      }
                    }

                    if (mts.getTaintedMembers().containsKey(sym.getName()) == true && mus.existSymbol(sym.getName()) == false)
                        sym.setTainted(1);                  
                  
                    if (inst != null){
                        SymbolTable t = inst.getInstanceAttributes();
                        if (t.containsSymbol(sym.getName()) || (scp.getEnclosingScope() != null && scp.getEnclosingScope().getScopeName().equals("this") && t.containsSymbol(sym.getName()))){
                            Scope scp_acc = (Scope) t.getSymbolOfListSymbols(sym.getName());
                            sym.setTainted(scp_acc.getScopeSymbol().getTainted());
                        }
                    }

                    scp.resolveSymbol(scp, sym, mts, mus, sym.getFileSymbol());
                    mts.removeDuplicates(sym);


                    if (csAtt_att != null){
                        inst.updateTainted(csAtt_att, sym);
                    }

                    if (csAtt_acc != null){
                        sym.setTainted(csAtt_acc.getScopeSymbol().getTainted());
                        scp.resolveSymbol(scp, sym, mts, mus, sym.getFileSymbol());
                    }

                    if (scp.getScopeSymbol().getIsSecureFunction() == false){
                        // se o scope e' tainted e e' uma funcao SQLI entao fazer correcao SQL
                        if (scp.getScopeSymbol().getIsTaintedFunction() == true && scp.getScopeSymbol().getTainted() == 1 && scp.getScopeSymbol().getIsVulDetected() == false){
                            correctVulner(scp, filename, GlobalDataXSS.MainSymbolTable, null, ms, mts, GlobalDataXSS.MainLinesToCorrect);
                        }
                                                
                    }
            }
            else{
                Scope scp1 = sym.getRootScope();
                if (scp1.getMembers().size() > 0){
                    FuncAndIncXSS css2 = new FuncAndIncXSS(scp1);
                    css2.resolveSymbolClassXSS(ms, scp1, mts, mus, filename, mtsc, inst, mct, ftime, csAtt_att, csAtt_acc, typeMeth, isClassInstruction);
                    scp.resolveSymbol(scp, sym, mts, mus, filename);
                    mts.removeDuplicates(sym);
                    return;
                }
                else{                
                ftime = false;
                
                // 1. obter instancia
                String instName = scp1.getEnclosingScope().getScopeName();
                String fileInst = scp1.getEnclosingScope().getScopeSymbol().getFileSymbol();
                InstanceTable instTable = GlobalDataXSS.MainInstancesTable.get(fileInst);
                SymbolTable st = GlobalDataXSS.MainSymbolTable.get(fileInst);
                String instName_aux = instName;
                Instance inst1 = null;
                if (st.getAliases().containsKey(instName_aux) == true)
                        instName_aux = st.getInstanceNameFromAliases(instName_aux, instTable);                
                
                if (instTable != null && instTable.containsInstance(instName_aux) == true)
                    inst1 = (Instance) instTable.getInstanceOfListInstances(instName_aux);

                // 2. obter classe
                String c_name = inst1.getInstanceClass();
                String cf_name = inst1.getFileOfClass();
                ClassTable ct = GlobalDataXSS.MainClassesTable.get(cf_name);
                ClassSymbol currentClass = ct.getClassOfListClasses(c_name);
                // 3. procurar metodo
                MethodClass mc = currentClass.getMethodClass(scp1.getScopeName(), "", mct);
                MethodSymbol mss_final = mc.getMeth();
                String mss_final_file = mc.getClassMeth().getFileClass();

                ret = null;
                ret = instanceMethod(mss_final, scp1, filename, mss_final_file, mts, mus, inst1, null, mct, "", ret, instTable, st);                

                if (ret != null && instTable.containsInstance(ret.getInstanceName()) == true){
                        Boolean a = true;
                        Scope enc_f = scp;
                        Scope enc_b;
                        while (a == true){
                                enc_b = enc_f.getEnclosingScope();
                                if (enc_b.getScopeName().equals(st.getScopeName()) == false)
                                        enc_f = enc_b;
                                else
                                        a = false;
                        }

                        Symbol sy = new Symbol(ret.getInstanceName(), ret.getTainted(), ret.getCreateLineInstance(), ret.getFileInstance());
                        sy.setIsClassInstruction(true);
                        st.getAliases().put(enc_f.getScopeName(), sy);
                        scp1.resolveSymbol(scp1, sym, mts, mus, sym.getFileSymbol());
                        mts.removeDuplicates(sym);

                        if (cs_att != null){
                                cs_att.getScopeSymbol().setTainted(ret.getTainted());
                                inst1.updateTainted(cs_att, sym);
                        }
                }            
                }
                scp.resolveSymbol(scp, sym, mts, mus, filename);
                mts.removeDuplicates(sym);         
           }
         }
         if (exit_for == true)
             break;
        }
        }
      }
    }
    }
    

    private static Instance instanceMethod(MethodSymbol ms, Scope scp, String filename, String file_methClass, TaintedTable mts, UntaintedTable mus, Instance inst, MethodSymbolCall mtsc, Map mct, String typeMeth, Instance objRet, InstanceTable instTab, SymbolTable symTab) throws IOException{          
        // ver quem e' o metodo
        String s = scp.getScopeName();
        String s1 = "", s2 = "";
        Instance ii = null, ret = null;

        //int i = -1;

        InstanceTable instances_table = null;
        if (instTab == null){
            if (typeMeth.equals("constructor") == true)
                instances_table = inst.getObjectsOfInstance();
            else
                instances_table = mtsc.getObjectsOfMethod();
        }
        else
            instances_table = instTab;

        Scope enc = scp.getEnclosingScope();
        String enc_s = enc.getScopeSymbol().getName();
        String instanceName = enc_s;
        if (enc.getScopeSymbol().getIsClassMethod() == false && objRet == null){
            // obj -> func()
            // ver se a instancia e' um clone
            if (mtsc.getAliases().containsKey(instanceName) == true){
                    instanceName = mtsc.getInstanceNameFromAliases(instanceName, instances_table);
            }

            if (instances_table.containsInstance(instanceName) == true){
                ii = (Instance) instances_table.getInstanceOfListInstances(instanceName);
                s1 = ii.getInstanceClass();
                s2 = ii.getFileOfClass();
                inst = ii;
            }
            else{
                // func() e' um metodo da instancia actual
                try{
                    s1 = inst.getInstanceClass();
                    s2 = inst.getFileOfClass();
                } catch (Exception e) {}
            }

            if (s1 != null){
                // obter classe
                ClassTable class_table = (ClassTable) mct.get(s2);
                ClassSymbol currentClass  = (ClassSymbol) class_table.getClassOfListClasses(s1);

                // obter metodo
                MethodClass mc = currentClass.getMethodClass(s, "", mct);
                MethodSymbol mss_final = mc.getMeth();
                String mss_final_file = mc.getClassMeth().getFileClass();

                /* Fazer a analise do metodo */
                if (mss_final != null){
                    MethodSymbolXSS meth_sym = new MethodSymbolXSS(mss_final);
                    int valueFinal = meth_sym.analyseMethodXSS(mss_final,scp.getScopeSymbol().getCodeLine(), scp, mts, mus, filename, mss_final_file, inst, mct, typeMeth);
                    scp.getScopeSymbol().setTainted(valueFinal);
                    ret = inst.getInstanceOfReturn(s, scp.getScopeSymbol().getCodeLine(), null, null, false);

                }
            }
        }
        else{
            // (obj -> func()) -> func()
            if (objRet != null){
                // obter classe
                s1 = objRet.getInstanceClass();
                s2 = objRet.getFileOfClass();
                ClassTable class_table = (ClassTable) mct.get(s2);
                ClassSymbol currentClass  = (ClassSymbol) class_table.getClassOfListClasses(s1);

                if (scp.getScopeSymbol().getIsClassMethod() == true){                
                    // obter metodo
                    MethodClass mc = currentClass.getMethodClass(s, "", mct);
                    MethodSymbol mss_final = mc.getMeth();
                    String mss_final_file = mc.getClassMeth().getFileClass();

                    /* Fazer a analise do metodo */
                    if (mss_final != null){
                        MethodSymbolXSS meth_sym = new MethodSymbolXSS(mss_final);
                        int valueFinal = meth_sym.analyseMethodXSS(mss_final,scp.getScopeSymbol().getCodeLine(), scp, mts, mus, filename, mss_final_file, inst, mct, typeMeth);
                        scp.getScopeSymbol().setTainted(valueFinal);
                        ret = inst.getInstanceOfReturn(s, scp.getScopeSymbol().getCodeLine(), null, null, false);
                    }
                }
                else{
                    if (scp.getScopeSymbol().getRootScope() == null){                       
                        FuncAndIncXSS css2 = new FuncAndIncXSS(scp);
                        css2.resolveSymbolClassXSS(null, scp, mts, mus, scp.getScopeSymbol().getFileSymbol(), mtsc, inst, mct, true, null, null, typeMeth, true);
                        scp.resolveSymbol(scp, (Symbol) scp.getMembers().get(0), mts, mus, scp.getScopeSymbol().getFileSymbol());
                        mts.removeDuplicates((Symbol) scp.getMembers().get(0));
                    }                
                }
            }
        }

        
        // ver se o metodo e' o ultimo a ser chamado
        Boolean endChain = true;
        Symbol sy = null;
        for (int j = 0; j < scp.getMembers().size(); j++){
            sy = (Symbol) scp.getMembers().get(j);
            if (sy.getRootScope() != null && sy.getIsParamUserFunction() == false){
                endChain = false;
                break;
            }
        }
        if (endChain == true){
            if (ret != null){
                ClassTable ct = (ClassTable) mct.get(ret.getFileOfClass());
                ClassSymbol ret_cl = (ClassSymbol) ct.getClassOfListClasses(ret.getInstanceClass());
                Instance r = new Instance(ret, ret_cl);
                instances_table.getMembers().add(r);
                scp.getScopeSymbol().setTainted(ret.getTainted());
                scp.resolveSymbol(scp, scp.getScopeSymbol(), mts, mus, scp.getScopeSymbol().getFileSymbol());
            }
        }
        else{
            ret = instanceMethod(ms, (Scope) sy, filename, file_methClass, mts, mus, inst, mtsc, mct, typeMeth, ret, instTab, symTab);
            scp.resolveSymbol(scp, sy, mts, mus, scp.getScopeSymbol().getFileSymbol());
            mts.removeDuplicates(sy);
        }

        return ret;
    }



    // Resolve symbols of include files that have yet analysed one time.
    // In this way, the include is added just one time in Scope Symtab_include
    // and on next times just need make data flow analysis to scope
    public void resolveSymbolIncludeXSS(Scope scp, TaintedTable mts, UntaintedTable mus, Map mift, String filename, Map mft, Map mst, Map mftt, Map MainLinesToCorrect) throws IOException{
         String filename_aux="";
        // Scope em analise e' um "include("...")"
        if (scp.getScopeName().equals("include") == true){
            Symbol s = (Symbol) scp.getMembers().get(0);
            String nameIncFile = s.getName();
            
            try{
                SymbolTable st = (SymbolTable) mift.get(nameIncFile);
                Iterator <Symbol> it = st.getMembers().iterator();
                Scope cs;
                Symbol sy;
                for(; it.hasNext();){
                    sy = it.next();
                    if (sy.getIsVariableSymbol() == false){
                        cs = (Scope) sy;
                        FuncAndIncXSS css = new FuncAndIncXSS(cs);
                        css.resolveSymbolIncludeXSS(cs, mts, mus, mift, nameIncFile, mft, mst, mftt, MainLinesToCorrect);
                    }
                }
            } catch (Exception e){} 
            //} catch (Exception e){System.out.println("File: "+ nameIncFile+" not exist");} 
        }
        else{
            if (scp.getScopeSymbol().getIsSecureFunction() == false){
                // verifica se o scope name e' uma chamada de funcao de UserFunction
                // primeiro nas funcoes definidas no file em analise
                MethodSymbolXSS mss_final = null;
                MethodSymbol mss = null;
                String mss_final_file = "";                
                if (mft.containsKey(filename) == true){
                    MethodTable mt = (MethodTable) mft.get(filename);
                    for(Iterator <MethodSymbol> it = mt.getMembers().iterator(); it.hasNext();){
                        mss = it.next();
                        if(mss.getFunctionName().equals(scp.getScopeName()) == true){
                            mss_final = new MethodSymbolXSS(mss);
                            mss_final_file = filename;
                            break;
                        }
                    }
                }
                if (mss_final == null){
                    // segundo nas UserFunctions definidas nos include files do file em analise
                    SymbolTable st = (SymbolTable) mift.get(filename);
                    if (st.getIncludeFiles().isEmpty() == false){
                        String s;
                        SymbolTable st1;
                        for(Iterator <String> it = st.getIncludeFiles().iterator(); it.hasNext();){
                            s = it.next();
                            st1 = (SymbolTable) mift.get(s);
                            if(st1.getHaveFunctions() == true){
                                MethodTable mt  = (MethodTable) mft.get(s);
                                for(Iterator <MethodSymbol> it1 = mt.getMembers().iterator(); it1.hasNext();){
                                    mss = it1.next();
                                    if(mss.getFunctionName().equals(scp.getScopeName()) == true){
                                        mss_final = new MethodSymbolXSS(mss);
                                        mss_final_file = s;
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }

                if (mss_final == null && scp.getScopeSymbol().getIsFunction() == true
                      && scp.getScopeSymbol().getIsTaintedFunction() == false
                      && scp.getScopeSymbol().getFileSymbol().equals(filename) == false){
                        // terceiro nas UserFunctions definidas nos include files do file que chamou a primeira
                        // funcao. Aqui estamos 'a procura de uma 2da funcao que esta' dentro de um include nao 
                        // pertencente ao file onde ela e' chamada, mas a um file que chama a 1 funcao
                        SymbolTable st = (SymbolTable) mst.get(filename);
                        if (st.getIncludeFiles().isEmpty() == false){
                            String s;
                            SymbolTable st1;
                            for(Iterator <String> it = st.getIncludeFiles().iterator(); it.hasNext();){
                                s = it.next();
                                st1 = (SymbolTable) mift.get(s);
                                if(st1.getHaveFunctions() == true){
                                    MethodTable mt  = (MethodTable) mft.get(s);
                                    for(Iterator <MethodSymbol> it1 = mt.getMembers().iterator(); it1.hasNext();){
                                        mss = it1.next();
                                        if(mss.getFunctionName().equals(scp.getScopeName()) == true){
                                            //mss_final = (MethodSymbolSqli) mss;
                                            mss_final = new MethodSymbolXSS(mss);
                                            mss_final_file = s;
                                            filename_aux = filename;
                                            filename= scp.getScopeSymbol().getFileSymbol();                                          
                                            break;
                                        }
                                    }
                                }
                                if (mss_final != null)
                                    break;
                            }
                        }
                        
                        // verifica nos outros includes. Para termos a certeza que se trata ou nao de uma userFunction
                        if (mss_final == null){
                            for(Iterator <SymbolTable> itt = mift.values().iterator(); itt.hasNext();){
                                SymbolTable st1 = itt.next();
                                if(st1.getHaveFunctions() == true && st.getIncludeFiles().contains(st1.getScopeName()) == false){
                                    MethodTable mt  = (MethodTable) mft.get(st1.getScopeName());
                                    for(Iterator <MethodSymbol> it1 = mt.getMembers().iterator(); it1.hasNext();){
                                        mss = it1.next();
                                        if(mss.getFunctionName().equals(scp.getScopeName()) == true){
                                            //mss_final = (MethodSymbolSqli) mss;
                                            mss_final = new MethodSymbolXSS(mss);
                                            mss_final_file = st1.getScopeName();
                                            filename_aux = filename;
                                            filename= scp.getScopeSymbol().getFileSymbol();                                          
                                            break;
                                        }
                                    }
                                }
                                if (mss_final != null)
                                    break;
                            }                        
                        }
                }                
                
                
                // se mss_final != null entao o scope name e' uma Userfunction
                // Fazer entao a analise na UserFunction
                if (mss_final != null){
                    String valueFinal = mss_final.analyseUserFunctionXSS((MethodSymbol) mss_final, scp.getScopeSymbol().getCodeLine(), scp, mts, mus, filename, mss_final_file, mift, mft, mst, mftt, MainLinesToCorrect);
                    if (filename_aux.isEmpty() == false){
                        filename = filename_aux;
                        filename_aux = "";
                    }                 
                }
                else{
                Symbol sym1, sym;
                Iterator <Symbol> it = scp.getMembers().iterator();
                for(;it.hasNext();){
                    sym = it.next();
                    if (sym.getRootScope() == null){
                            if (mts.getTaintedMembers().containsKey(sym.getName()) == true && mus.existSymbol(sym.getName()) == false)
                                sym.setTainted(1);

                            scp.resolveSymbol(scp, sym, mts, mus, sym.getFileSymbol());
                            mts.removeDuplicates(sym);

                            // se o scope e' tainted e e' uma funcao XSS entao fazer correcao
                            if ((scp.getScopeSymbol().getIsTaintedFunction() == true && scp.getScopeSymbol().getTainted() == 1 && scp.getScopeSymbol().getIsVulDetected() == false)){
                                correctVulner(scp, filename, mst, mift, null, mts, MainLinesToCorrect);
                            }                            
                            
                    }
                    else{
                        Scope scp1 = (Scope) sym.getRootScope();
                        FuncAndIncXSS css1 = new FuncAndIncXSS(scp1);
                        css1.resolveSymbolIncludeXSS((Scope) scp1, mts, mus, mift, filename, mft, mst, mftt, MainLinesToCorrect);
                        scp.resolveSymbol(scp, sym, mts, mus, filename);
                        mts.removeDuplicates(sym);
                        
                        if ((scp.getScopeSymbol().getIsTaintedFunction() == true && scp.getScopeSymbol().getTainted() == 1 && scp.getScopeSymbol().getIsVulDetected() == false)){
                            correctVulner(scp, filename, mst, mift, null, mts, MainLinesToCorrect);
                        }
                    }  
               }
            }
        }
      }
    }




    // Resolve symbols of function defined by user that have yet analysed one time.
    // In this way, the function is added just one time in Scope Symtab_functions
    // and on next times just need make data flow analysis to scope
    public void resolveSymbolFunctionXSS(MethodSymbol ms, Scope scp, TaintedTable mts, UntaintedTable mus, Map mift, String filename, Map mft, Map mst, Map mftt, Map MainLinesToCorrect) throws IOException{       
        String filename_aux="";
        if (scp.getScopeName().equals("include") == true){
            Symbol s = (Symbol) scp.getMembers().get(0);
            String nameIncFile = s.getName();
            String x = nameIncFile.substring(1,nameIncFile.length()-1);
            SymbolTable st = (SymbolTable) mift.get(x);
            Iterator <Symbol> it = st.getMembers().iterator();
            Scope cs;
            for(; it.hasNext();){
		cs = (Scope) it.next();
                FuncAndIncXSS css = new FuncAndIncXSS(cs);
                css.resolveSymbolIncludeXSS((Scope) cs, mts, mus, mift, x, mft, mst, mftt, MainLinesToCorrect);
            }
            return;
        }
            if (scp.getScopeSymbol().getIsSecureFunction() == false){
            // verifica se o scope name e' uma chamada de funcao de UserFunction
            // primeiro nas funcoes definidas no file em analise
            MethodSymbolXSS mss_final = null;
            MethodSymbol mss = null;
            String mss_final_file = "";
            if (scp.getScopeSymbol().getIsUserFunction() == true){
            if (mft.containsKey(filename) == true){
                MethodTable mt = (MethodTable) mft.get(filename);
                for(Iterator <MethodSymbol> it = mt.getMembers().iterator(); it.hasNext();){
                    mss = it.next();
                    if(mss.getFunctionName().equals(scp.getScopeName()) == true){
                        mss_final = new MethodSymbolXSS(mss);
                        mss_final_file = filename;
                        break;
                    }
                }
            }
            if (mss_final == null){
                // segundo nas UserFunctions definidas nos include files do file em analise
                SymbolTable st = (SymbolTable) mst.get(filename);
                if (st.getIncludeFiles().isEmpty() == false){
                    String s;
                    SymbolTable st1;
                    for(Iterator <String> it = st.getIncludeFiles().iterator(); it.hasNext();){
                        s = it.next();
			st1 = (SymbolTable) mift.get(s);
			if(st1.getHaveFunctions() == true){
                            MethodTable mt  = (MethodTable) mft.get(s);
			    for(Iterator <MethodSymbol> it1 = mt.getMembers().iterator(); it1.hasNext();){
				mss = it1.next();
				if(mss.getFunctionName().equals(scp.getScopeName()) == true){
                                    mss_final = new MethodSymbolXSS(mss);
                                    mss_final_file = s;
                                    break;
				}
                            }
			}
                        if (mss_final != null)
                            break;
                    }
                }
            }
            }

                if (mss_final == null && scp.getScopeSymbol().getIsFunction() == true
                      && scp.getScopeSymbol().getIsTaintedFunction() == false
                      && scp.getScopeSymbol().getFileSymbol().equals(filename) == false){
                        // terceiro nas UserFunctions definidas nos include files do file que chamou a primeira
                        // funcao. Aqui estamos 'a procura de uma 2da funcao que esta' dentro de um include nao 
                        // pertencente ao file onde ela e' chamada, mas a um file que chama a 1 funcao
                        SymbolTable st = (SymbolTable) mst.get(filename);
                        if (st.getIncludeFiles().isEmpty() == false){
                            String s;
                            SymbolTable st1;
                            for(Iterator <String> it = st.getIncludeFiles().iterator(); it.hasNext();){
                                s = it.next();
                                st1 = (SymbolTable) mift.get(s);
                                if(st1.getHaveFunctions() == true){
                                    MethodTable mt  = (MethodTable) mft.get(s);
                                    for(Iterator <MethodSymbol> it1 = mt.getMembers().iterator(); it1.hasNext();){
                                        mss = it1.next();
                                        if(mss.getFunctionName().equals(scp.getScopeName()) == true){
                                            mss_final = new MethodSymbolXSS(mss);
                                            mss_final_file = s;
                                            filename_aux = filename;
                                            filename= scp.getScopeSymbol().getFileSymbol();                                          
                                            break;
                                        }
                                    }
                                }
                                if (mss_final != null)
                                    break;
                            }
                        }
                        
                        // verifica nos outros includes. Para termos a certeza que se trata ou nao de uma userFunction
                        if (mss_final == null){
                            for(Iterator <SymbolTable> itt = mift.values().iterator(); itt.hasNext();){
                                SymbolTable st1 = itt.next();
                                if(st1.getHaveFunctions() == true && st.getIncludeFiles().contains(st1.getScopeName()) == false){
                                    MethodTable mt  = (MethodTable) mft.get(st1.getScopeName());
                                    for(Iterator <MethodSymbol> it1 = mt.getMembers().iterator(); it1.hasNext();){
                                        mss = it1.next();
                                        if(mss.getFunctionName().equals(scp.getScopeName()) == true){
                                            mss_final = new MethodSymbolXSS(mss);
                                            mss_final_file = st1.getScopeName();
                                            filename_aux = filename;
                                            filename= scp.getScopeSymbol().getFileSymbol();                                          
                                            break;
                                        }
                                    }
                                }
                                if (mss_final != null)
                                    break;
                            }                        
                        }
                }            
            

            // Fazer entao a analise na UserFunction
            if (mss_final != null){
                String valueFinal = mss_final.analyseUserFunctionXSS(mss, scp.getScopeSymbol().getCodeLine(), scp, mts, mus, filename, mss_final_file, mift, mft, mst, mftt, MainLinesToCorrect);
                if (filename_aux.isEmpty() == false){
                    filename = filename_aux;
                    filename_aux = "";
                }
            }
            else{
            Symbol sym1, sym;           
            
            // Se o nome do scope for o nome de argumento da chamada da funcao entao o nome do scope muda para ele
            if(ms.getArgsFunction().containsKey(scp.getScopeName()) == true){
                ArgFunctionSymbol arg = (ArgFunctionSymbol) ms.getArgsFunction().get(scp.getScopeName());
                scp.getScopeSymbol().setName(arg.getCallFunctionArg());
                return;
            }
                       
            Iterator <Symbol> it = scp.getMembers().iterator();
            for(;it.hasNext();){
                    sym = it.next();
                    if (sym.getRootScope() == null){
                            if (ms.getArgsFunction().containsKey(sym.getName()) == true){
                                ArgFunctionSymbol arg = (ArgFunctionSymbol) ms.getArgsFunction().get(sym.getName());
                                sym.setName(arg.getCallFunctionArg());
                            }
                            
                            if (mts.getTaintedMembers().containsKey(sym.getName()) == true && mus.existSymbol(sym.getName()) == false)
                                sym.setTainted(1);
                            
                            scp.resolveSymbol(scp, sym, mts, mus, sym.getFileSymbol());
                            mts.removeDuplicates(sym);

                            // se o scope e' tainted e e' uma funcao XSS entao fazer correcao
                                if ((scp.getScopeSymbol().getIsTaintedFunction() == true && scp.getScopeSymbol().getTainted() == 1 && scp.getScopeSymbol().getIsVulDetected() == false)){
                                    correctVulner(scp, filename, mst, null, ms, mts, MainLinesToCorrect);
                                }
                            
                    }
                    else{
                        Scope scp1 = sym.getRootScope();
                        FuncAndIncXSS css1 = new FuncAndIncXSS(scp1);
                        css1.resolveSymbolFunctionXSS(ms, scp1, mts, mus, mift, filename, mft, mst, mftt, MainLinesToCorrect);
                        scp.resolveSymbol(scp, sym, mts, mus, filename);
                        mts.removeDuplicates(sym);
                        
                        if ((scp.getScopeSymbol().getIsTaintedFunction() == true && scp.getScopeSymbol().getTainted() == 1 && scp.getScopeSymbol().getIsVulDetected() == false)){
                            correctVulner(scp, filename, mst, null, ms, mts, MainLinesToCorrect);
                        }
                    }
            }              
            
            Scope aux = scp.getScopeSymbol().getRootScope();
            if (aux.getScopeName().equals("return") == true && aux.getScopeSymbol().getTainted() == 1){
                ms.getScopeSymbol().setTainted(1);
     
                //inserir ms como dependente do return
                RootTaintedSymbol Rrt;
                Rrt = (RootTaintedSymbol) mts.getTaintedMembers().get("return");
                RelatedTaintedSymbol rtt;
                rtt =   (RelatedTaintedSymbol) Rrt.getListTaintedMembers().get(Rrt.getListTaintedMembers().size()-1);
                // Caso o ms nao pertenca 'a lista da funcao tainted
                if (rtt.getTaintedMembers().containsKey(ms.getScopeName()) == false){
                    VariableTaintedSymbol var = new VariableTaintedSymbol(ms.getScopeName(),
                                                ms.getScopeSymbol().getCodeLine(), ms.getScopeSymbol().getTainted(),
                                                ms.getScopeSymbol().getFileSymbol());
                    rtt.define(var);
                }
                else{
                    VariableTaintedSymbol var = (VariableTaintedSymbol) rtt.getTaintedMembers().get(ms.getScopeName());
                    var.InsertLine(ms.getScopeSymbol().getCodeLine());
                    var.InsertFile(ms.getScopeSymbol().getFileSymbol());
                }
            }
        }
      }
}

    // Resolve symbols of SymbolTable.
    public void resolveSymbolXSS(Scope scp, TaintedTable mts, UntaintedTable mus, Map mift, String filename, Map mft, Map mst, Map mftt, Map MainLinesToCorrect, Map mct, Map mobjt) throws IOException{
        if (scp.getScopeName().equals("include")){
            try{
                this.resolveSymbolIncludeXSS(scp, mts, mus, mift, filename, mft, mst, mftt, MainLinesToCorrect);
            } catch (Exception e){
                //System.out.println("Error processing include file: "+this.getName());
            }
            return;
        }
        
        if (scp.getScopeSymbol().getIsSecureFunction() == false){
            if (scp.getScopeSymbol().getIsUserFunction() == true){
                try{
                    this.resolveSymbolFunctionXSS(null, scp, mts, mus, mift, filename, mft, mst, mftt, MainLinesToCorrect);
                } catch (Exception e){
                //System.out.println("Error processing include file: "+this.getName());
                }
                return;
            }

            if(scp.getScopeSymbol().getIsClassInstruction() == true){
                try{
                    this.resolveSymbolClassXSS(null, scp, mts, mus, filename, null, null, mct, true, null, null, filename, true);
                } catch (Exception e){
                //System.out.println("Error processing include file: "+this.getName());
                }
                return;
            }

                     
            // Analise de scope normal
            Symbol sym1, sym;
            Iterator <Symbol> it = scp.getMembers().iterator();
            for(;it.hasNext();){
                sym = it.next();
                if (sym.getRootScope() == null){
                    if (mts.getTaintedMembers().containsKey(sym.getName()) == true && mus.existSymbol(sym.getName()) == false)
                        sym.setTainted(1);

                    scp.resolveSymbol(scp, sym, mts, mus, sym.getFileSymbol());
                    mts.removeDuplicates(sym);

                    // se o scope e' tainted e e' uma funcao SQLI entao fazer correcao XSS
                    if ((scp.getScopeSymbol().getIsTaintedFunction() == true && scp.getScopeSymbol().getTainted() == 1 && scp.getScopeSymbol().getIsVulDetected() == false)){
                            correctVulner(scp, filename, mst, null, null, mts, MainLinesToCorrect);
                    }                     
                }
                else{
                    Scope scp1 = (Scope) sym.getRootScope();
                    FuncAndIncXSS css1 = new FuncAndIncXSS(scp1);
                    css1.resolveSymbolXSS(sym.getRootScope(), mts, mus, mift, filename, mft, mst, mftt, MainLinesToCorrect, mct, mobjt);
                    scp.resolveSymbol(scp, sym, mts, mus, filename);
                    mts.removeDuplicates(sym);
                    
                    // se o scope e' tainted e e' uma funcao SQLI entao fazer correcao XSS
                    if ((scp.getScopeSymbol().getIsTaintedFunction() == true && scp.getScopeSymbol().getTainted() == 1 && scp.getScopeSymbol().getIsVulDetected() == false)){
                            correctVulner(scp, filename, mst, null, null, mts, MainLinesToCorrect);
                    }
                }
           }            
        }
    }
    
private void correctVulner(Scope scp, String filename, Map mst, Map mift, MethodSymbol ms, TaintedTable mts, Map MainLinesToCorrect) throws IOException{        
        scp.getScopeSymbol().setIsVulDetected(true);
        SymbolTable currentSymbolTable = new SymbolTable(scp.getScopeSymbol().getFileSymbol());
        SymbolTable ct;          
        
        int m = 0;
        if (mift == null || mift.size() == 0)
            m = 0;

        if (ms == null){
            if (m == 0){
                ct = (SymbolTable) mst.get(scp.getScopeSymbol().getFileSymbol());
                currentSymbolTable.setAllSymbols(ct.getMembers());
                currentSymbolTable.setAllIncludeFiles(ct.getIncludeFiles());
            }
            else{    
                ct = (SymbolTable) mift.get(scp.getScopeSymbol().getFileSymbol());
                currentSymbolTable.setAllSymbols(ct.getMembers());
                currentSymbolTable.setAllIncludeFiles(ct.getIncludeFiles());
            }
        }
        else
            currentSymbolTable.setAllSymbols(ms.getMembers());
        VulnerabilityPathXSS vvp = new VulnerabilityPathXSS(scp);


/* **************
* CORRECAO e DM
*/                        
        VulnerXSS vv = vvp.collectVulnPathXSS(GlobalDataXSS.UserInput, mts, currentSymbolTable, false, MainLinesToCorrect);

        /*
                DATA MINING
        */

        int fp = vv.getVectorAttributesOfVulner().predictFalsePositive(true);
        vv.setIsFP(fp);          

        if (GlobalDataXSS.MainNumFP.containsKey(vv.getFileOf_SS()) == false)
            GlobalDataXSS.MainNumFP.put(vv.getFileOf_SS(), 0);

        if (GlobalDataXSS.MainNumVul.containsKey(vv.getFileOf_SS()) == false)
            GlobalDataXSS.MainNumVul.put(vv.getFileOf_SS(), 0);            

        //try{
        if (vv.IsFP() == 1){                          
            int nFP = GlobalDataXSS.MainNumFP.get(vv.getFileOf_SS()) + 1;
            GlobalDataXSS.MainNumFP.put(vv.getFileOf_SS(), nFP);                            
        }
        else{
            if (scp.getScopeSymbol().getTainted() == 1){                     
                int nVul = GlobalDataXSS.MainNumVul.get(vv.getFileOf_SS()) + 1;
                GlobalDataXSS.MainNumVul.put(vv.getFileOf_SS(), nVul);
            }

            /* Correccao */
            AutoCorrectXSS acs = new AutoCorrectXSS(vv);
            acs.buildCorrect(MainLinesToCorrect);
        }
        //} catch (Exception e) {}



        GlobalDataXSS.currentVulner = null;            
    //}
    }        
    
    @Override
    public List<Symbol> getMembers() {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    public void print(Scope scp) {
        throw new UnsupportedOperationException("Not supported yet.");
    }




}
