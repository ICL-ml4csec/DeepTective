/*
 * Classe que permite verificar o sistema operativo que está a ser
 * executado a aplicacao awap
 */
package org.homeunix.wap.utils;

/**
 *
 * @author iberiam
 */
public class OSValidator {
	public static boolean isWindows() {
 
		String os = System.getProperty("os.name").toLowerCase();
		// windows
		return (os.indexOf("win") >= 0);
	}
 
	public static boolean isMac() {
 		String os = System.getProperty("os.name").toLowerCase();
		// Mac
		return (os.indexOf("mac") >= 0);
 	}
 
	public static boolean isUnix() {
 		String os = System.getProperty("os.name").toLowerCase();
		// linux or unix
		return (os.indexOf("nix") >= 0 || os.indexOf("nux") >= 0);
 	}
 
	public static boolean isSolaris() {
 		String os = System.getProperty("os.name").toLowerCase();
		// Solaris
		return (os.indexOf("sunos") >= 0);
 	}
        
	public static String windowsVersion() {
 
		String version = System.getProperty("os.version").toLowerCase();
		// windows
		return version;
	}        
}
